import fire
import json
from pathlib import Path

PROJECT_DIRNAME = Path('./').resolve()


def load_notebook_json(path: Path) -> dict:
    with open(path) as fo:
        data = json.load(fo)
    return data


def get_cell_code(cell) -> str:
    source_code = ''.join(cell['source'])
    return source_code


def get_export_cells(notebook_json):
    export_cells = []
    for cell in notebook_json['cells']:
        cell_type = cell['cell_type']
        source_code = get_cell_code(cell)
        if cell_type == 'code' and source_code.startswith('#export\n'):
            export_cells.append(cell)
    return export_cells


def get_export_filename(notebook_path: Path) -> Path:
    name = notebook_path.name
    digits = name.split('_')[0]
    export_path = PROJECT_DIRNAME / f'exports/exp_{digits}.py'
    return export_path


def save_cells(cells, path):
    print(f'Saving {len(cells)} cells to {path}')
    with open(path, 'w') as fw:
        for cell in cells:
            code = get_cell_code(cell)
            fw.write(code)
            fw.write('\n')
    return

def main(path: str) -> None:
    path = Path(path)
    notebook_data = load_notebook_json(path)
    cells = get_export_cells(notebook_data)
    export_filename = get_export_filename(path)
    save_cells(cells, export_filename)
    return


if __name__ == '__main__':
    fire.Fire(main)
