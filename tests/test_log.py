import os
import subprocess
import sys

import pytest

# Log at every level from inside each package's namespace (loguru names records by
# the calling module), after importing the packages in `imports`.
SCRIPT = """
import sys
from loguru import logger
for name in sys.argv[2].split(","):
    __import__(name)
code = "for lv in ('TRACE', 'DEBUG', 'INFO'): logger.log(lv, f'{__name__}:{lv}')"
for package in sys.argv[1].split(","):
    exec(code, {"logger": logger, "__name__": package + ".module"})
"""


def run(env_levels, packages="patchology", imports="patchology", setup=""):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_LOG_LEVEL")}
    env.update(env_levels)
    proc = subprocess.run(
        [sys.executable, "-c", setup + SCRIPT, packages, imports],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.stderr


def shown(err, package):
    return [lv for lv in ("TRACE", "DEBUG", "INFO") if f"{package}.module:{lv}" in err]


@pytest.mark.parametrize(
    "level, expected",
    [
        (None, ["INFO"]),
        ("DEBUG", ["DEBUG", "INFO"]),
        ("debug", ["DEBUG", "INFO"]),
        ("ALL", ["TRACE", "DEBUG", "INFO"]),
    ],
)
def test_level_from_env(level, expected):
    env = {} if level is None else {"PATCHOLOGY_LOG_LEVEL": level}
    assert shown(run(env), "patchology") == expected


def test_levels_are_per_package():
    env = {"PATCHOLOGY_LOG_LEVEL": "INFO", "OTHERPKG_LOG_LEVEL": "DEBUG"}
    err = run(env, packages="patchology,otherpkg,app")
    assert shown(err, "patchology") == ["INFO"]
    assert shown(err, "otherpkg") == ["DEBUG", "INFO"]
    assert shown(err, "app") == ["INFO"]  # no variable: INFO


def test_unknown_level_fails_at_import():
    assert "does not exist" in run({"PATCHOLOGY_LOG_LEVEL": "LOUD"})


def test_application_logging_is_left_alone():
    # The application removed the default handler and logs everything itself.
    setup = (
        "from loguru import logger; import sys; logger.remove(); logger.add(sys.stderr, level=0)\n"
    )
    assert shown(run({}, setup=setup), "patchology") == ["TRACE", "DEBUG", "INFO"]
