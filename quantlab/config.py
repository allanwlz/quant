"""Runtime settings from the environment, with quant/prd/quant.env as the default source."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT.parent / 'prd' / 'quant.env'


def _load_env_file(path):
    """Fill missing variables from a KEY=VALUE file; real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and key and not key.startswith('#'):
            os.environ.setdefault(key, value.strip().strip('"'))


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    tushare_token: str
    tickflow_token: str

    @property
    def db_path(self):
        return self.data_dir / 'market.duckdb'


def load(env_file=ENV_FILE):
    _load_env_file(Path(os.getenv('QUANT_ENV_FILE', env_file)))
    data_dir = Path(os.getenv('QUANT_DATA_DIR', ROOT.parent / 'prd' / 'data'))
    return Settings(data_dir=data_dir,
                    tushare_token=os.getenv('QUANT_TUSHARE_TOKEN', ''),
                    tickflow_token=os.getenv('QUANT_TICKFLOW_TOKEN', ''))
