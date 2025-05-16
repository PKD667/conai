import unittest

from typed_lambda_parser import (
    tokenize, Parser, Token,
    TypeVariable, FunctionType,
    Variable, Abstraction, Application,
    ParserSyntaxError  # Changed from SyntaxError
)

class TestTypedLambdaParser(unittest.TestCase):

    def assertParsesTo(self, code, expected_str):
        tokens = tokenize(code)
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertEqual(str(ast), expected_str)

    def assertTypeParsesTo(self, type_code, expected_str):
        tokens = tokenize(type_code)
        # parse_type might not consume all tokens if it's part of a larger expression,
        # so we add an EOF token manually for isolated type parsing.
        parser = Parser(tokens[:-1] + [Token('EOF', '')]) # Replace original EOF if any, or add one
        parsed_type = parser.parse_type()
        self.assertEqual(str(parsed_type), expected_str)
        # Ensure all tokens were consumed for type parsing
        self.assertEqual(parser.current_token().type, 'EOF', f"Not all tokens consumed for type: {type_code}")


    # --- Tokenizer Tests ---
    def test_tokenize_simple_identifiers(self):
        self.assertEqual(tokenize("x y z"), [
            Token('IDENTIFIER', 'x'), Token('IDENTIFIER', 'y'), Token('IDENTIFIER', 'z'), Token('EOF', '')
        ])

    def test_tokenize_lambda_syntax(self):
        self.assertEqual(tokenize("λx:T. x"), [
            Token('LAMBDA', 'λ'), Token('IDENTIFIER', 'x'), Token('COLON', ':'),
            Token('IDENTIFIER', 'T'), Token('DOT', '.'), Token('IDENTIFIER', 'x'), Token('EOF', '')
        ])
        self.assertEqual(tokenize("\\x:T. x"), [
            Token('LAMBDA', '\\'), Token('IDENTIFIER', 'x'), Token('COLON', ':'),
            Token('IDENTIFIER', 'T'), Token('DOT', '.'), Token('IDENTIFIER', 'x'), Token('EOF', '')
        ])

    def test_tokenize_arrows_parens(self):
        self.assertEqual(tokenize("(A -> B)"), [
            Token('LPAREN', '('), Token('IDENTIFIER', 'A'), Token('ARROW', '->'),
            Token('IDENTIFIER', 'B'), Token('RPAREN', ')'), Token('EOF', '')
        ])

    def test_tokenize_whitespace(self):
        self.assertEqual(tokenize("  x  \n y "), [
            Token('IDENTIFIER', 'x'), Token('IDENTIFIER', 'y'), Token('EOF', '')
        ])

    def test_tokenize_mismatch(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected character: @"):
            tokenize("@")

    # --- Type Parsing Tests ---
    def test_parse_type_variable(self):
        self.assertTypeParsesTo("T", "T")

    def test_parse_function_type_simple(self):
        self.assertTypeParsesTo("A -> B", "A -> B")

    def test_parse_function_type_right_associative(self):
        self.assertTypeParsesTo("A -> B -> C", "A -> (B -> C)")

    def test_parse_function_type_with_parens(self):
        self.assertTypeParsesTo("(A -> B) -> C", "(A -> B) -> C")

    def test_parse_type_nested_parens(self):
        self.assertTypeParsesTo("((T))", "T")
        self.assertTypeParsesTo("(A -> (B -> C)) -> D", "(A -> (B -> C)) -> D")

    def test_parse_type_invalid_start(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='ARROW'.* in type, expected IDENTIFIER or LPAREN"):
            tokens = tokenize("-> A")
            Parser(tokens).parse_type()

    def test_parse_type_missing_arrow_rhs(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='EOF'.* in type, expected IDENTIFIER or LPAREN"):
            tokens = tokenize("A -> ")
            Parser(tokens).parse_type()
            
    def test_parse_type_unclosed_paren(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Expected token RPAREN but got EOF"):
            tokens = tokenize("(A")
            Parser(tokens).parse_type()

    # --- Expression Parsing Tests ---
    def test_parse_variable(self):
        self.assertParsesTo("x", "x")

    def test_parse_abstraction(self):
        self.assertParsesTo("λx:T. x", "(λx:T. x)")
        self.assertParsesTo("\\x:A->B. y", "(λx:(A -> B). y)")

    def test_parse_application_simple(self):
        self.assertParsesTo("f x", "(f x)")

    def test_parse_application_left_associative(self):
        self.assertParsesTo("f x y", "((f x) y)")

    def test_parse_application_with_abstraction(self):
        self.assertParsesTo("(λx:T. x) y", "((λx:T. x) y)")
        self.assertParsesTo("y (λx:T. x)", "(y (λx:T. x))")

    def test_parse_nested_abstraction(self):
        self.assertParsesTo("λf:A->B. λx:A. f x", "(λf:(A -> B). (λx:A. (f x)))")

    def test_parse_parenthesized_expression(self):
        self.assertParsesTo("(x)", "x")
        self.assertParsesTo("(f x)", "(f x)")
        self.assertParsesTo("((f x) y)", "((f x) y)")

    def test_parse_complex_expression(self):
        self.assertParsesTo("((λx:T. x) (λy:U. y)) z", "(((λx:T. x) (λy:U. y)) z)")
        self.assertParsesTo("λf:(A->A). λx:A. f (f x)", "(λf:(A -> A). (λx:A. (f (f x))))")

    def test_parse_expression_with_complex_types(self):
        self.assertParsesTo("λx:(A -> B) -> C. x", "(λx:((A -> B) -> C). x)")
        self.assertParsesTo("λx:A -> B -> C. x", "(λx:(A -> (B -> C)). x)") # Type is right-associative
    
    # --- Error Handling Tests ---
    def test_parse_error_unexpected_eof(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Expected token IDENTIFIER but got EOF"):
            Parser(tokenize("λ")).parse() # Missing param name

        with self.assertRaisesRegex(ParserSyntaxError, "Expected token COLON but got EOF"):
            Parser(tokenize("λx")).parse() # Missing colon

        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='EOF'.* in type, expected IDENTIFIER or LPAREN"):
            Parser(tokenize("λx:")).parse() # Missing type

        with self.assertRaisesRegex(ParserSyntaxError, "Expected token DOT but got EOF"):
            Parser(tokenize("λx:T")).parse() # Missing dot

        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='EOF'.*expected IDENTIFIER or LPAREN for an atom"):
            Parser(tokenize("λx:T.")).parse() # Missing body

    def test_parse_error_mismatched_parens(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Expected token RPAREN but got EOF"):
            Parser(tokenize("(x")).parse()
        
        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='RPAREN'.* at end of expression"):
            Parser(tokenize("x)")).parse()

    def test_parse_error_incomplete_abstraction(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Expected token IDENTIFIER but got DOT"):
            Parser(tokenize("λ.x")).parse()

    def test_parse_error_atom_expected(self):
        # For "f -> x", 'f' is parsed as an atom. Then '->' is an unexpected token at the end of the expression.
        with self.assertRaisesRegex(ParserSyntaxError, r"Unexpected token Token\(type='ARROW', value='->'\) at end of expression"):
            Parser(tokenize("f -> x")).parse() # Application cannot start with arrow

    def test_parse_error_unexpected_token_at_end(self):
        # (λx:T.x)y is a valid application: ((λx:T. x) y)
        self.assertParsesTo("(λx:T.x)y", "((λx:T. x) y)")

        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='LAMBDA'.* at end of expression"):
            Parser(tokenize("x λy:U.y")).parse() # This is parsed as (x) followed by λ...

    def test_parse_type_in_abstraction_error(self):
        with self.assertRaisesRegex(ParserSyntaxError, "Unexpected token Token.*type='DOT'.* in type, expected IDENTIFIER or LPAREN"):
            Parser(tokenize("λx:A->.x")).parse() # Incomplete type in abstraction

if __name__ == '__main__':
    unittest.main()
