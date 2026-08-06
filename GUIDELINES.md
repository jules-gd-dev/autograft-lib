# AutoGraft Development Guidelines

## Git Workflow
- We work ONLY on the `dev` branch for features. `main` is for stable releases.
- Commits must be done for every logical change.
- We use Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

## Code Quality Rules
- **Line count:** Maximum 100 to 150 lines per Python file. If a file exceeds 150 lines, it MUST be split into separate modules.
- **Linting:** Code must pass `pylint` with a score of 9.0+.
- **Testing:** We use `pytest`. Minimum code coverage is 97%.
- **Typing:** Python type hints are mandatory for all function arguments and return types.
