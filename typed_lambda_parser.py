import re
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional

# --- AST Node Definitions ---

@dataclass
class TypeVariable:
    name: str

    def __str__(self):
        return self.name

@dataclass
class FunctionType:
    param_type: 'Type'
    return_type: 'Type'

    def __str__(self):
        param_str = str(self.param_type)
        if isinstance(self.param_type, FunctionType):
            param_str = f"({param_str})"
        
        return_type_str = str(self.return_type)
        if isinstance(self.return_type, FunctionType): # Ensure return type that is a function type is parenthesized
            return_type_str = f"({return_type_str})"
            
        return f"{param_str} -> {return_type_str}"

Type = Union[TypeVariable, FunctionType]

@dataclass
class Variable:
    name: str

    def __str__(self):
        return self.name

@dataclass
class Abstraction:
    param_name: str
    param_type: Type
    body: 'Expression'

    def __str__(self):
        param_type_str = str(self.param_type)
        if isinstance(self.param_type, FunctionType): # Ensure param_type that is a function type is parenthesized
            param_type_str = f"({param_type_str})"
        return f"(λ{self.param_name}:{param_type_str}. {self.body})"

@dataclass
class Application:
    func: 'Expression'
    arg: 'Expression'

    def __str__(self):
        return f"({self.func} {self.arg})"

Expression = Union[Variable, Abstraction, Application]

# --- Tokenizer ---

# Define a custom exception for parsing errors to avoid conflict with built-in SyntaxError
class ParserSyntaxError(Exception):
    pass

TOKEN_SPECIFICATION = [
    ('LAMBDA',     r'[λ\\]'),
    ('ARROW',      r'->'),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('COLON',      r':'),
    ('DOT',        r'\.'),
    ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('WHITESPACE', r'\s+'),
    ('MISMATCH',   r'.'),
]

TOKEN_REGEX = re.compile('|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in TOKEN_SPECIFICATION))

@dataclass
class Token:
    type: str
    value: str

def tokenize(code: str) -> List[Token]:
    tokens = []
    for mo in TOKEN_REGEX.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'WHITESPACE':
            continue
        elif kind == 'MISMATCH':
            raise ParserSyntaxError(f"Unexpected character: {value}")
        tokens.append(Token(kind, value))
    tokens.append(Token('EOF', '')) # End of File token
    return tokens

# --- Parser ---

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        self.pos += 1

    def expect(self, token_type: str) -> Token:
        token = self.current_token()
        if token and token.type == token_type:
            self.advance()
            return token
        expected_value = token_type
        actual_value = token.type if token else "EOF"
        raise ParserSyntaxError(f"Expected token {expected_value} but got {actual_value}")

    def parse(self) -> Expression:
        expr = self.parse_expression()
        if self.current_token() and self.current_token().type != 'EOF':
            raise ParserSyntaxError(f"Unexpected token {self.current_token()} at end of expression")
        return expr

    # Type ::= BaseType { '->' BaseType }
    # BaseType ::= IDENTIFIER | '(' Type ')'
    def parse_type(self) -> Type:
        left_type = self.parse_base_type()
        while self.current_token() and self.current_token().type == 'ARROW':
            self.expect('ARROW')
            right_type = self.parse_base_type()
            left_type = FunctionType(left_type, right_type) # Right associative for A -> B -> C means A -> (B -> C)
                                                            # but our structure FunctionType(param, return) naturally builds this
                                                            # if we parse T1 -> T2 -> T3 as (T1 -> (T2 -> T3))
                                                            # The loop structure T_left = Func(T_left, T_right) makes it left-associative.
                                                            # We need to parse it as T1 -> (parse_type_rhs())
        return left_type

    def parse_type_recursive(self) -> Type: # Corrected for right-associativity
        t1 = self.parse_base_type()
        if self.current_token() and self.current_token().type == 'ARROW':
            self.expect('ARROW')
            t2 = self.parse_type_recursive() # Recurse for the right part
            return FunctionType(t1, t2)
        return t1
    
    # Re-aliasing for clarity in the main parse method
    parse_type = parse_type_recursive

    def parse_base_type(self) -> Type:
        token = self.current_token()
        if token.type == 'IDENTIFIER':
            self.advance()
            return TypeVariable(token.value)
        elif token.type == 'LPAREN':
            self.expect('LPAREN')
            type_expr = self.parse_type()
            self.expect('RPAREN')
            return type_expr
        else:
            raise ParserSyntaxError(f"Unexpected token {token} in type, expected IDENTIFIER or LPAREN")

    # Expr ::= Abstraction | Application
    def parse_expression(self) -> Expression:
        token = self.current_token()
        if token.type == 'LAMBDA':
            return self.parse_abstraction()
        else:
            return self.parse_application()

    # Abstraction ::= (λ | \) IDENTIFIER ':' Type '.' Expr
    def parse_abstraction(self) -> Abstraction:
        self.expect('LAMBDA')
        param_name_token = self.expect('IDENTIFIER')
        self.expect('COLON')
        param_type = self.parse_type()
        self.expect('DOT')
        body = self.parse_expression()
        return Abstraction(param_name_token.value, param_type, body)

    # Application ::= Atom { Atom } (left-associative)
    def parse_application(self) -> Expression:
        expr = self.parse_atom()
        while True:
            # Check if the next token could start an atom (for application)
            # This handles cases like `x y z` which is `((x y) z)`
            # or `(λx:A.x) y`
            # We stop if we see EOF, RPAREN, DOT, COLON, ARROW, LAMBDA
            # (tokens that cannot start an atom or are delimiters for other constructs)
            next_token = self.current_token()
            if next_token and next_token.type in ('IDENTIFIER', 'LPAREN'):
                arg = self.parse_atom()
                expr = Application(expr, arg)
            else:
                break
        return expr

    # Atom ::= IDENTIFIER | '(' Expr ')'
    def parse_atom(self) -> Expression:
        token = self.current_token()
        if token.type == 'IDENTIFIER':
            self.advance()
            return Variable(token.value)
        elif token.type == 'LPAREN':
            self.expect('LPAREN')
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        else:
            raise ParserSyntaxError(f"Unexpected token {token}, expected IDENTIFIER or LPAREN for an atom")

# --- Example Usage ---
if __name__ == '__main__':
    test_expressions = [
        "x",
        "λx:T. x",
        "λf:A->B. λx:A. f x",
        "(λx:T. x) y",
        "f x y", # equivalent to (f x) y
        "λx:(A -> B) -> C. x",
        "λx:A -> B -> C. x", # equivalent to λx:(A -> (B -> C)). x
        "((λx:T. x) (λy:U. y)) z",
        "λf:(A->A). λx:A. f (f x)",
        "(λx: (A -> B). x) (λy: A. y)", # Type in parens
        "λx: ( (A -> B) -> C ). x" # Complex type in parens
    ]

    for expr_str in test_expressions:
        print(f"Parsing: {expr_str}")
        try:
            tokens = tokenize(expr_str)
            # print(f"Tokens: {tokens}")
            parser = Parser(tokens)
            ast = parser.parse()
            print(f"AST: {ast}")
            print(f"Stringified: {str(ast)}")
        except ParserSyntaxError as e:
            print(f"Syntax Error: {e}")
        print("-" * 20)

    # Test type parsing specifically
    test_types = [
        "A",
        "A -> B",
        "A -> B -> C", # A -> (B -> C)
        "(A -> B) -> C",
        "((A)) -> B"
    ]
    print("\n--- Type Parsing Tests ---")
    for type_str in test_types:
        print(f"Parsing type: {type_str}")
        try:
            tokens = tokenize(type_str + " ") # Add space for EOF if needed by tokenizer logic
            # Filter out EOF if it's not expected by parse_type directly
            tokens_for_type = [t for t in tokens if t.type != 'EOF']
            
            parser = Parser(tokens_for_type + [Token('EOF', '')]) # Add EOF for parser's current_token logic
            parsed_type = parser.parse_type()
            print(f"Parsed Type: {parsed_type}")
            print(f"Stringified Type: {str(parsed_type)}")
        except ParserSyntaxError as e:
            print(f"Syntax Error: {e}")
        print("-" * 20)

