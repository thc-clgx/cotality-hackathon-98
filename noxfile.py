import nox


@nox.session(python=["3.12", "3.13"])
def lint(session: nox.Session):
    session.install("ruff")
    session.run("ruff", "check", ".", external=True)


@nox.session(python=["3.12", "3.13"])
def format_check(session: nox.Session):
    session.run("ruff", "format", ".", "--check", external=True)


@nox.session(python=["3.12", "3.13"])
def format_code(session: nox.Session):
    session.run("ruff", "format", ".", external=True)


@nox.session(python=["3.12", "3.13"])
def type_check(session: nox.Session):
    session.install("mypy")
    session.run("mypy", "src/python_project_template")
