import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(import_path, '../'))
sys.path.append(os.path.join(import_path, '../TVAE/generate'))
sys.path.append(os.getcwd())

import old_module
sys.modules['allennlp'] = sys.modules['old_module']
sys.modules['allennlp.modules'] = sys.modules['old_module']
sys.modules['allennlp.modules.feedforward'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper'] = sys.modules['old_module']

import argparse
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()
import pickle as pi
import pandas as pd
import random
from predict import main
from pipeline.classifier import Classifier
from GOLEM.cocrystal_processing import run_experiment, molecule_search_setup, get_final_dataset

#путь до файла с предсгенерированными SMILES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_GAN_PATH = os.path.join(BASE_DIR, 'database_GAN.csv')
#DATABASE_GAN_PATH = os.path.join(BASE_DIR, 'database_CCDC.csv')

classification = Classifier()

# Функция для сохранения молекулы в sdf-файл
def create_sdf_file(molecule, file_name):
    mol = Chem.MolFromSmiles(molecule)
    writer = Chem.SDWriter(file_name)
    writer.write(mol)
    writer.close()

def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)

def get_mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)

def parameter():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--drug', type=str, default='CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    parser.add_argument('--num_molecules', type=int, default=100)
    parser.add_argument('--optimization', type=bool, default=True, choices=[True, False])
    parser.add_argument('--properties', type=str, nargs='+', default=['unobstructed', 'orthogonal_planes', 'h_bond_bridging'])
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parameter())
    drug = args['drug']
    num_molecules_to_generate = args['num_molecules']
    optimization = args['optimization']
    properties = args['properties']

    # Загрузка предсгенерированных SMILES из файла DATABASE_GAN_PATH
    df_smiles = pd.read_csv(DATABASE_GAN_PATH)
    smiles_list = df_smiles['smiles'].tolist()

    # Классификация и выбор коформеров по заданным свойствам
    df = classification.clf_results(drug, smiles_list, properties)

    # Создание директории для коформеров, если её нет
    path_coformer = os.path.join("pipeline", "coformers")
    if not os.path.isdir(path_coformer):
        os.mkdir(path_coformer)

    ######################################################################################################
    # Запуск эволюционной оптимизации (если включена)
    if optimization:
        print("Эволюционная оптимизация запущена...")
        init_path = os.path.join("pipeline", "coformers", f"mols_{drug}.csv")
        df.to_csv(init_path)
        print(f"Исходные данные сохранены в {init_path}")

        col_name = 'generated_coformers'
        results_folder = os.path.join("pipeline", "results_golem")
        exp_ids = []
        exp_id = run_experiment(molecule_search_setup,
                                initial_data_path=init_path,
                                col_name=col_name,
                                max_heavy_atoms=50,
                                trial_timeout=60,
                                trial_iterations=5,
                                pop_size=20,
                                visualize=True,
                                num_trials=2,
                                drug=drug,
                                result_dir=results_folder)
        exp_ids.append(exp_id)
        print(f"Оптимизация завершена, получен идентификатор эксперимента: {exp_id}")
        
        exp_results_folder = os.path.join(results_folder, exp_id)
        df = get_final_dataset(exp_results_folder, drug)
        print("Результаты эволюционной оптимизации (первые 5 строк):")
        print(df.head())
    else:
        print("Эволюционная оптимизация не включена.")
