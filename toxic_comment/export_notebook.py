import json
from pathlib import Path


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


def save_cells(cells, path):
    print(f'Saving {len(cells)} cells to {path}')
    with open(path, 'w') as fw:
        for cell in cells:
            code = get_cell_code(cell)
            fw.write(code)
            fw.write('\n')
    return

if __name__ == '__main__':
    path = 'notebooks/00_baseline.ipynb'
    notebook_data = load_notebook_json(path)
    cells = get_export_cells(notebook_data)
    save_cells(cells, 'exports/00.py')
