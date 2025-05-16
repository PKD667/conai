import re
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional, Dict
import copy

# --- AST Node Definitions ---

@dataclass
class TypeVariable:
    name: str

    def __str__(self):
        return self.name
    
    def visualize(self,d=1) -> str:
        # show in tree form
        return f"TypeVariable({self.name})"

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
    
    def visualize(self,d=1) -> str:
        # show in tree form
        return f"FunctionType(\n {' '*d} {self.param_type.visualize(d=d+1)}, \n {' '*d} {self.return_type.visualize(d=d+1)})"



Type = Union[TypeVariable, FunctionType]

@dataclass
class Variable:
    name: str

    def __str__(self):
        return self.name
    
    def visualize(self,d=1) -> str:
        # show in tree form
        return f"Variable({self.name})"

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

    def visualize(self, d=1) -> str:
        # show in tree form
        return f"Abstraction(\n {' '*d} {self.param_name}, \n {' '*d} {self.param_type.visualize(d=d+1)}, \n {' '*d} {self.body.visualize(d=d+1)})"
    

@dataclass
class Application:
    func: 'Expression'
    arg: 'Expression'

    def __str__(self):
        return f"({self.func} {self.arg})"
    
    def visualize(self, d=1) -> str:
        # show in tree form
        return f"Application(\n {' '*d} {self.func.visualize(d=d+1)}, \n {' '*d} {self.arg.visualize(d=d+1)})"

Expression = Union[Variable, Abstraction, Application]

# --- Type Checking ---

class TypeCheckError(Exception):
    """Custom exception for type checking errors."""
    pass

TypeEnvironment = Dict[str, Type]

def infer_type(expression: Expression, context: TypeEnvironment) -> Type:
    """
    Infers the type of an expression given a type environment (context).
    Raises TypeCheckError if a type error is found.
    """
    if isinstance(expression, Variable):
        if expression.name in context:
            return context[expression.name]
        else:
            raise TypeCheckError(f"TypeCheckError: Free variable '{expression.name}' not found in context {context}.")
    
    elif isinstance(expression, Abstraction):
        # Create a new context for the body of the abstraction
        new_context = context.copy() # Use copy to avoid modifying the outer context
        new_context[expression.param_name] = expression.param_type
        
        body_type = infer_type(expression.body, new_context)
        return FunctionType(expression.param_type, body_type)
        
    elif isinstance(expression, Application):
        func_type = infer_type(expression.func, context)
        arg_type = infer_type(expression.arg, context)
        
        if not isinstance(func_type, FunctionType):
            raise TypeCheckError(
                f"TypeCheckError: Cannot apply non-function type '{func_type}' (from '{expression.func}') "
                f"to argument '{expression.arg}' of type '{arg_type}'."
            )
        
        # Check if the argument type matches the function's parameter type
        # The __eq__ method of TypeVariable and FunctionType handles structural equality.
        if func_type.param_type == arg_type:
            return func_type.return_type
        else:
            raise TypeCheckError(
                f"TypeCheckError: Type mismatch in application '{expression}'. "
                f"Function '{expression.func}' expects argument of type '{func_type.param_type}', "
                f"but received argument '{expression.arg}' of type '{arg_type}'."
            )
            
    else:
        # This case should ideally not be reached if Expression is correctly typed
        raise TypeCheckError(f"TypeCheckError: Unknown expression type encountered: {type(expression)}")

# --- Tokenizer ---

# Define a custom exception for parsing errors to avoid conflict with built-in SyntaxError
class ParserSyntaxError(Exception):
    def __init__(self, message, parser_instance=None):
        super().__init__(message)
        self.parser_instance = parser_instance # Store the parser instance if available

TOKEN_SPECIFICATION = [
    # recognize Unicode λ, backslash \, or the literal "lambda"
    ('LAMBDA',     r'(?:λ|\\|lambda)'),
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
            raise ParserSyntaxError(f"Unexpected character: {value}") # No parser instance here
        tokens.append(Token(kind, value))
    tokens.append(Token('EOF', '')) # End of File token
    return tokens

# --- Parser ---

class Parser:
    def __init__(self, tokens: List[Token], debug: bool = False):
        self.tokens = tokens
        self.pos = 0
        self.debug = debug
        if self.debug:
            print(f"Parser initialized with tokens: {self.tokens}")

    def _log(self, message: str):
        if self.debug:
            print(f"[DEBUG Parser] Pos {self.pos}: {message}")

    def current_token(self) -> Optional[Token]:
        token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
        return token

    def advance(self):
        token = self.current_token()
        self._log(f"advance() from {token}")
        self.pos += 1
        self._log(f"          to {self.current_token()}")

    def expect(self, token_type: str) -> Token:
        token = self.current_token()
        self._log(f"expect({token_type}) - current: {token}")
        if token and token.type == token_type:
            self.advance()
            self._log(f"expect({token_type}) - matched: {token}")
            return token
        expected_value = token_type
        actual_value = token.type if token else "EOF"
        self._log(f"expect({token_type}) - FAILED. Expected {expected_value}, got {actual_value}")
        raise ParserSyntaxError(f"Expected token {expected_value} but got {actual_value}", parser_instance=self)

    def parse(self) -> Expression:
        self._log("parse() called")
        expr = self.parse_expression()
        current = self.current_token()
        if current and current.type != 'EOF':
            self._log(f"parse() - Unexpected token {current} at end of expression")
            raise ParserSyntaxError(f"Unexpected token {current} at end of expression", parser_instance=self)
        self._log(f"parse() -> {expr}")
        return expr

    def parse_type(self) -> Type:
        self._log(f"parse_type() called. Current token: {self.current_token()}")
        left_type = self.parse_base_type()
        self._log(f"parse_type() - parsed base_type: {left_type}. Current token: {self.current_token()}")
        if self.current_token() and self.current_token().type == 'ARROW':
            self.expect('ARROW')
            self._log(f"parse_type() - found ARROW, parsing RHS. Current token: {self.current_token()}")
            right_type = self.parse_type()
            result = FunctionType(left_type, right_type)
            self._log(f"parse_type() -> {result} (FunctionType)")
            return result
        self._log(f"parse_type() -> {left_type} (BaseType)")
        return left_type

    def parse_base_type(self) -> Type:
        token = self.current_token()
        self._log(f"parse_base_type() called. Current token: {token}")
        if token.type == 'IDENTIFIER':
            self.advance()
            result = TypeVariable(token.value)
            self._log(f"parse_base_type() -> {result} (TypeVariable: {token.value})")
            return result
        elif token.type == 'LPAREN':
            self.expect('LPAREN')
            self._log(f"parse_base_type() - found LPAREN, parsing inner type. Current token: {self.current_token()}")
            type_expr = self.parse_type()
            self.expect('RPAREN')
            self._log(f"parse_base_type() -> {type_expr} (Parenthesized Type)")
            return type_expr
        else:
            self._log(f"parse_base_type() - FAILED. Unexpected token {token}")
            raise ParserSyntaxError(f"Unexpected token {token} in type, expected IDENTIFIER or LPAREN", parser_instance=self)

    def parse_expression(self) -> Expression:
        token = self.current_token()
        self._log(f"parse_expression() called. Current token: {token}")
        if token.type == 'LAMBDA':
            result = self.parse_abstraction()
            self._log(f"parse_expression() -> {result} (Abstraction)")
            return result
        else:
            result = self.parse_application()
            self._log(f"parse_expression() -> {result} (Application)")
            return result

    def parse_abstraction(self) -> Abstraction:
        self._log(f"parse_abstraction() called. Current token: {self.current_token()}")
        self.expect('LAMBDA')
        param_name_token = self.expect('IDENTIFIER')
        self._log(f"parse_abstraction() - param_name: {param_name_token.value}")
        self.expect('COLON')
        param_type = self.parse_type()
        self._log(f"parse_abstraction() - param_type: {param_type}")
        self.expect('DOT')
        body = self.parse_expression()
        self._log(f"parse_abstraction() - body: {body}")
        result = Abstraction(param_name_token.value, param_type, body)
        self._log(f"parse_abstraction() -> {result}")
        return result

    def parse_application(self) -> Expression:
        self._log(f"parse_application() called. Current token: {self.current_token()}")
        expr = self.parse_atom()
        self._log(f"parse_application() - parsed first atom: {expr}. Current token: {self.current_token()}")
        while True:
            next_token = self.current_token()
            self._log(f"parse_application() - in loop, next_token: {next_token}")
            if next_token and next_token.type in ('IDENTIFIER', 'LPAREN'):
                self._log(f"parse_application() - applying to another atom. Current token: {self.current_token()}")
                arg = self.parse_atom()
                expr = Application(expr, arg)
                self._log(f"parse_application() - new expr after application: {expr}. Current token: {self.current_token()}")
            else:
                self._log(f"parse_application() - no more atoms to apply, breaking loop.")
                break
        self._log(f"parse_application() -> {expr}")
        return expr

    def parse_atom(self) -> Expression:
        token = self.current_token()
        self._log(f"parse_atom() called. Current token: {token}")
        if token.type == 'IDENTIFIER':
            self.advance()
            result = Variable(token.value)
            self._log(f"parse_atom() -> {result} (Variable: {token.value})")
            return result
        elif token.type == 'LPAREN':
            self.expect('LPAREN')
            self._log(f"parse_atom() - found LPAREN, parsing inner expression. Current token: {self.current_token()}")
            expr = self.parse_expression()
            self.expect('RPAREN')
            self._log(f"parse_atom() -> {expr} (Parenthesized Expression)")
            return expr
        else:
            self._log(f"parse_atom() - FAILED. Unexpected token {token}")
            raise ParserSyntaxError(f"Unexpected token {token}, expected IDENTIFIER or LPAREN for an atom", parser_instance=self)

# --- Example Usage ---
if __name__ == '__main__':
    DEBUG_PARSER = False # Set to True to enable parser debugging
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
            parser = Parser(tokens, debug=DEBUG_PARSER)
            ast = parser.parse()
            print(f"AST: {ast.visualize()}")
            print(f"Stringified: {str(ast)}")

            # Perform type checking
            inferred_type = infer_type(ast, {}) # Empty context for closed terms
            print(f"Inferred Type: {inferred_type}")

        except ParserSyntaxError as e:
            print(f"Syntax Error: {e}")
        except TypeCheckError as e:
            print(f"Type Check Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
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
            tokens = tokenize(type_str + ' ') # Add space for EOF if needed by tokenizer logic
            tokens_for_type = [t for t in tokens if t.type != 'EOF']
            
            parser = Parser(tokens_for_type + [Token('EOF', '')], debug=DEBUG_PARSER)
            parsed_type = parser.parse_type()
            print(f"Parsed Type: {parsed_type}")
            print(f"Stringified Type: {str(parsed_type)}")
        except ParserSyntaxError as e:
            print(f"Syntax Error: {e}")
        print("-" * 20)

