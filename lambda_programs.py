from typed_lambda_parser import tokenize, Parser, ParserSyntaxError, infer_type, TypeCheckError, TypeEnvironment, TypeVariable
from beta_reducer import reduce_to_normal_form

# Helper function to process and display lambda programs
def process_lambda_program(name: str, lambda_str: str, description: str = "", reduce_it: bool = False, max_steps: int = 20):
    """
    Parses a lambda expression string, type checks it, prints its AST and type, 
    and optionally reduces it.
    """
    print(f"\n--- {name} ---")
    if description:
        print(f"Description: {description}")
    print(f"Lambda expression: {lambda_str}")
    
    try:
        tokens = tokenize(lambda_str)
        parser = Parser(tokens, debug=False) 
        ast = parser.parse()
        
        print(f"Parsed AST (visualized):\n{ast.visualize()}")
        print(f"Stringified AST: {str(ast)}")

        # Type checking
        try:
            initial_context: TypeEnvironment = {}
            temp_context: TypeEnvironment = {}
            if "first_arg" in lambda_str: temp_context["first_arg"] = TypeVariable("A_arg") # Placeholder type
            if "second_arg" in lambda_str: temp_context["second_arg"] = TypeVariable("B_arg") # Placeholder type
            if "z_var" in lambda_str: temp_context["z_var"] = TypeVariable("A") # Consistent with (λy:A. y) z_var

            inferred_type_val = infer_type(ast, temp_context)
            print(f"Inferred Type: {inferred_type_val}")
            print(f"This type corresponds to the proposition proven by the lambda term.")

        except TypeCheckError as tce:
            print(f"Type Check Error: {tce}")

        if reduce_it:
            print("\nAttempting to reduce:")
            normal_form, history = reduce_to_normal_form(ast, max_steps=max_steps)
            
            if len(history) > 1:
                print("Reduction steps:")
                for i, step_expr in enumerate(history):
                    print(f"  {'Start:' if i == 0 else 'Step '+str(i)+':'} {step_expr}")
            else:
                is_normal = True
                if len(history) == 1:
                    _, changed_at_first_try = reduce_to_normal_form(ast, max_steps=1)
                    if len(changed_at_first_try) > 1:
                         is_normal = False

                if is_normal and not (len(history) > 1 and history[0] == history[-1]):
                     print("Expression is already in normal form.")
                elif not (len(history) > 1 and history[0] == history[-1]):
                     print("No reduction occurred within the first step or expression is already normal.")

            print(f"Result after reduction attempt: {normal_form}")
            if len(history) > max_steps:
                 print(f"Note: Reduction stopped after {max_steps} steps.")

    except ParserSyntaxError as e:
        print(f"Syntax Error: {e}")
    except TypeCheckError as e:
        print(f"Type Check Error during processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40)

if __name__ == '__main__':
    print("=" * 40)
    print("      Simple Lambda Calculus Programs")
    print("=" * 40)

    process_lambda_program(
        name="Identity Function",
        lambda_str="λx:A. x",
        description="A function that returns its input. Type: A -> A."
    )

    process_lambda_program(
        name="Constant Function (K Combinator)",
        lambda_str="λx:A. λy:B. x",
        description="A function that takes two arguments and returns the first. Type: A -> B -> A."
    )

    process_lambda_program(
        name="Apply Function (Eta-expansion)",
        lambda_str="λf:(A->B). λx:A. f x",
        description="Takes a function and an argument, and applies the function to the argument. Type: (A->B) -> A -> B."
    )

    process_lambda_program(
        name="Function Composition",
        lambda_str="λf:(B->C). λg:(A->B). λx:A. f (g x)",
        description="Takes two functions f and g, and an argument x, returns f(g(x)). Type: (B->C) -> (A->B) -> A -> C."
    )

    process_lambda_program(
        name="Argument Swapping Combinator",
        lambda_str="λf:(A->B->C). λy:B. λx:A. f x y",
        description="Takes a function f (of type A->B->C) and arguments y (type B), x (type A), and applies them as f x y. Type: (A->B->C) -> B -> A -> C."
    )

    print("\n" + "=" * 40)
    print(" Curry-Howard Isomorphism Examples (Proofs)")
    print("=" * 40)

    process_lambda_program(
        name="Proof of A → A",
        lambda_str="λx:A. x",
        description="Proposition: A implies A. The identity function serves as a proof. If we assume A (x:A), we can produce A (x)."
    )

    process_lambda_program(
        name="Proof of (A → B) → A → B (Modus Ponens)",
        lambda_str="λf:(A->B). λx:A. f x",
        description="Proposition: (A implies B) implies (A implies B). Given a proof of (A implies B) (f) and a proof of A (x), we can produce a proof of B (f x)."
    )

    process_lambda_program(
        name="Proof of A → (B → A)",
        lambda_str="λx:A. λy:B. x",
        description="Proposition: A implies (B implies A). If we assume A (x), then for any assumption B (y), we can still produce A (x)."
    )

    process_lambda_program(
        name="Proof of (B → C) → (A → B) → A → C (Transitivity of Implication)",
        lambda_str="λf:(B->C). λg:(A->B). λx:A. f (g x)",
        description="Proposition: ((B implies C) and (A implies B)) implies (A implies C). This is function composition."
    )
    
    process_lambda_program(
        name="Proof of (A → B → C) → B → A → C (Commutation of Assumptions)",
        lambda_str="λh:(A->(B->C)). λy:B. λx:A. h x y", # Note: A->B->C is A->(B->C)
        description="Proposition: (A implies (B implies C)) implies (B implies (A implies C))."
    )

    process_lambda_program(
        name="Proof of A → ((A → B) → B) (Modus Ponens, alternative form)",
        lambda_str="λx:A. λf:(A->B). f x",
        description="Proposition: A implies ((A implies B) implies B). Given A (x) and (A implies B) (f), we can produce B (f x)."
    )

    print("\n" + "=" * 40)
    print("      Beta Reduction Example")
    print("=" * 40)
    
    process_lambda_program(
        name="Composition of Identity Functions",
        lambda_str="((λf:(T->T). λg:(T->T). λx:T. f (g x)) (λy:T. y)) (λz:T. z)",
        description="Applying the composition function to two identity functions. Expected to reduce to λx:T. x.",
        reduce_it=True
    )

    process_lambda_program(
        name="K Combinator Application",
        lambda_str="((λx:A. λy:B. x) first_arg) second_arg",
        description="Applying the K combinator. Expected to reduce to first_arg. Type of 'first_arg' is A_arg, type of 'second_arg' is B_arg. Result type: A_arg.",
        reduce_it=True
    )

    process_lambda_program(
        name="Self-application like (Church Numeral TWO)",
        lambda_str="(λf:(A->A). λx:A. f (f x)) (λy:A. y) z_var", # Represents (Two Id) z_var
        description="Represents applying the Church numeral Two to an identity-like function and then to a variable. Type of z_var is A. Result type: A.",
        reduce_it=True
    )
