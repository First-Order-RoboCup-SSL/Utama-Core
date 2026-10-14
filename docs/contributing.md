# Contributing

How to set up an editor, commit, and get a pull request merged. Quick start and repository
layout: the [README](../README.md). Context for coding agents: [`AGENTS.md`](../AGENTS.md).

## pixi

- `pixi install` installs the environment, including every package with an `__init__.py`:
  run it again after adding one.
- `pixi shell` enters the environment; `pixi run <task>` runs a task from `pixi.toml` without
  entering it (see [tools.md](tools.md#pixi-tasks)).
- For a one-off run of a module, `pixi run python -m path.to.your_file` (`/` replaced by `.`,
  no `.py`).
- In VS Code, if the run button picks the wrong environment (`robosim`), choose the interpreter
  by hand: `Ctrl + Shift + P` → `Select Interpreter`.

## Code style

- Type every variable.
- Format with Black. In VS Code, install the `Black Formatter` extension, open
  `Open User Settings (JSON)` from the command palette and add to the `"[python]"` field:

  ```json
  "[python]": {
      "editor.defaultFormatter": "ms-python.black-formatter",
      "editor.formatOnSave": true
  }
  ```

- `pixi run lint` runs the whole pre-commit stack (black, ruff, isort) on every file;
  `pixi run test` runs the tests. Run both before calling a change done.

## Commits

1. Each feature lives on its own branch. Clear out stale branches.
2. Run `pixi run precommit-install` once, so the pre-commit checks run on every commit.
3. If a commit fails the pre-commit checks, the hook has often already fixed the files: look at
   the output (`Open Git Log` in VS Code's popup), stage the changes and commit again.
4. On Windows the popup may show `bash: warning: setlocale: LC_ALL: cannot change locale
   (en_US.UTF-8)`. That is not the failure, just the first warning in the output. To silence it:

   ```bash
   sudo apt-get update
   sudo apt-get install -y locales
   sudo locale-gen en_US.UTF-8
   sudo update-locale LANG=en_US.UTF-8
   source ~/.bashrc
   ```

## Pull requests

A pull request into `main` needs:

1. a `release:major`, `release:minor` or `release:patch` label: every merge to `main` releases
   a new version automatically (`.github/workflows/release.yml`);
2. passing CI, tests and lint;
3. to be up to date with `main`:

   ```bash
   git checkout main
   git pull
   git checkout <your_branch>
   git merge main
   ```

4. every Copilot comment reviewed (not all must be addressed: Copilot makes mistakes too);
5. an approval from an assigned reviewer.

A strategy branch (`strategy/*`) is also checked by `tools/check_strategy_branch.py`: it must not
change the evaluation or the opponents it is scored against.

## `CLAUDE.md` and `AGENTS.md`

`CLAUDE.md` is a symlink to `AGENTS.md`, the single source of agent context (`AGENTS.md` is the
cross-vendor default; Claude Code reads `CLAUDE.md`). Edit `AGENTS.md`, never the symlink.
Linux, macOS and WSL check it out correctly. On native Windows, git only creates symlinks with
Developer Mode or Administrator privileges; without them `CLAUDE.md` arrives as a 9-byte text
file, fixed by `git config --global core.symlinks true && git checkout -- CLAUDE.md`.
