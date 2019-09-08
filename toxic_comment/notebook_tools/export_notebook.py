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
        if cell_type == 'code' and source_code.startswith('#export'):
            export_cells.append(cell)
    return export_cells


def get_export_filename(notebook_path: Path) -> Path:
    name = notebook_path.name
    digits = name.split('_')[0]
    export_path = PROJECT_DIRNAME / f'exports/exp_{digits}.py'
    return export_path


def get_cell_path(cell):
    first_line = cell['source'][0]
    if first_line.strip().endswith('.py'):
        ending = first_line.split()[1].strip()
        return PROJECT_DIRNAME / f'exports/{ending}'
    return None


def accumulate_cells(cells) -> dict:
    where_to_save = dict()
    for cell in cells:
        save_path = get_cell_path(cell)
        where_to_save[save_path] = where_to_save.get(save_path, []) + [cell]
    return where_to_save


def save_cells(cells, path):
    save_data = accumulate_cells(cells)
    save_data[str(path)] = save_data.pop(None,[])
    for path, cells in save_data.items():
        with open(path, 'w') as fw:
            code = '\n'.join([get_cell_code(cell) for cell in cells])
            fw.write(code)
    return

    print(f'Saving {len(cells)} cells to {path}')
    for cell in cells:
        code = get_cell_code(cell)
        save_path = get_cell_path(cell)
        if save_path is None: save_path = path
        print(save_path)
        with open(path, 'w+') as fw:
            fw.write(code)
            fw.write('\n')
        
    #with open(path, 'w') as fw:
    #    for cell in cells:
    #        code = get_cell_code(cell)
    #        fw.write(code)
    #        fw.write('\n')
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
