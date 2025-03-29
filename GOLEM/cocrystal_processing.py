import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Type, Optional, Sequence, List, Iterable, Callable

from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.operators.selection import tournament_selection
from golem.core.optimisers.opt_history_objects.individual import Individual

from GOLEM.analysis import get_final_dataset, visualize_results, get_statistical_significance, \
    plot_experiment_comparison
from GOLEM.scripts.moead_selection import MOEAD

import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)

import numpy as np
import pandas as pd
from golem.core.dag.verification_rules import has_no_self_cycled_nodes, has_no_isolated_components, \
    has_no_isolated_nodes
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.visualisation.opt_viz_extra import visualise_pareto
from golem.core.optimisers.adaptive.agent_trainer import AgentTrainer
from golem.core.optimisers.adaptive.history_collector import HistoryReader
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

from GOLEM.scripts.mol_adapter import MolAdapter
from GOLEM.scripts.mol_advisor import MolChangeAdvisor
from GOLEM.scripts.mol_graph import MolGraph
from GOLEM.scripts.mol_graph_parameters import MolGraphRequirements
from GOLEM.scripts.mol_metrics import CocrystalsMetrics
from GOLEM.scripts.mol_mutations import CHEMICAL_MUTATIONS
from GOLEM.scripts.utils import project_root

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))


def get_methane() -> MolGraph:
    methane = 'C'
    return MolGraph.from_smiles(methane)


def pretrain_agent(optimizer: EvoGraphOptimizer, objective: Objective, results_dir: str) -> AgentTrainer:
    agent = optimizer.mutation.agent
    trainer = AgentTrainer(objective, optimizer.mutation, agent)
    # load histories
    history_reader = HistoryReader(Path(results_dir))
    # train agent
    trainer.fit(histories=history_reader.load_histories(), validate_each=100)
    return trainer


def molecule_search_setup(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                          adaptive_kind: MutationAgentTypeEnum = MutationAgentTypeEnum.random,
                          max_heavy_atoms: int = 50,
                          atom_types: Optional[List[str]] = None,
                          bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                          timeout: Optional[timedelta] = None,
                          num_iterations: Optional[int] = None,
                          pop_size: int = 20,
                          drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2',
                          initial_molecules: Optional[Sequence[MolGraph]] = None):
    metrics = CocrystalsMetrics(drug)
    objective = Objective(
        quality_metrics={'orthogonal_planes': metrics.orthogonal_planes,
                         'unobstructed': metrics.unobstructed,
                         'h_bond_bridging': metrics.h_bond_bridging},
        is_multi_objective=True
    )
    evaluator = MultiprocessingDispatcher(adapter=MolAdapter(), n_jobs=-1).dispatch(objective)
    init_pop = [Individual(MolAdapter().adapt(graph)) for graph in initial_molecules]
    evaluator(init_pop)

    requirements = MolGraphRequirements(
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        bond_types=bond_types,
        early_stopping_timeout=np.inf,
        early_stopping_iterations=np.inf,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        keep_history=True,
        n_jobs=-1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        pop_size=pop_size,
        max_pop_size=pop_size,
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        elitism_type=ElitismTypesEnum.replace_worst,
        mutation_types=CHEMICAL_MUTATIONS,
        crossover_types=[CrossoverTypesEnum.none],
        selection_types=MOEAD(tournament_selection, init_pop, 2),
        adaptive_mutation_type=adaptive_kind
    )
    graph_gen_params = GraphGenerationParams(
        adapter=MolAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes, has_no_isolated_components, has_no_isolated_nodes],
        advisor=MolChangeAdvisor(),
    )

    initial_graphs = initial_molecules or [get_methane()]
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


def run_experiment(optimizer_setup: Callable,
                   optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                   adaptive_kind: MutationAgentTypeEnum = MutationAgentTypeEnum.random,
                   max_heavy_atoms: int = 50,
                   atom_types: Optional[List[str]] = None,
                   bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                   initial_data_path: Optional[str] = None,
                   col_name: str = 'generated_coformers',
                   pop_size: int = 20,
                   num_trials: int = 1,
                   trial_timeout: Optional[int] = None,
                   trial_iterations: Optional[int] = None,
                   visualize: bool = False,
                   save_history: bool = True,
                   drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2',
                   result_dir: Optional[str] = None,
                   pretrain_dir: Optional[str] = None
                   ):
    optimizer_id = optimizer_cls.__name__.lower()[:3]
    experiment_id = f'Experiment [optimizer={optimizer_id} pop_size={pop_size}]'
    exp_name = f'init_{Path(initial_data_path).stem}_{optimizer_id}_{adaptive_kind.value}_popsize{pop_size}_min{trial_timeout}'
    result_dir = Path(result_dir) / exp_name

    initial_smiles = pd.read_csv(initial_data_path)[col_name]
    initial_molecules = []
    for smiles in initial_smiles:
        try:
            mol = MolGraph.from_smiles(smiles)
            initial_molecules.append(mol)
        except Exception as ex:
            print(ex)
            continue

    trial_results = []
    trial_histories = []
    trial_timedelta = timedelta(minutes=trial_timeout) if trial_timeout else None

    optimizer, objective = optimizer_setup(optimizer_cls,
                                           adaptive_kind,
                                           max_heavy_atoms,
                                           atom_types,
                                           bond_types,
                                           trial_timedelta,
                                           trial_iterations,
                                           pop_size,
                                           drug,
                                           initial_molecules)
    if pretrain_dir:
        pretrain_agent(optimizer, objective, pretrain_dir)
    for trial in range(num_trials):

        found_graphs = optimizer.optimise(objective)
        history = optimizer.history

        if visualize:
            molecules = [MolAdapter().restore(graph) for graph in found_graphs]
            save_dir = Path(result_dir) / 'visualisations' / f'trial_{trial}'
            visualize_run_results(set(molecules), objective, history, save_dir)
        if save_history:
            result_dir.mkdir(parents=True, exist_ok=True)
            history.save(result_dir / f'history_trial_{trial}.json')
        trial_results.extend(history.final_choices)
        trial_histories.append(history)

    # Compute mean & std for metrics of trials
    ff = objective.format_fitness
    trial_metrics = np.array([ind.fitness.values for ind in trial_results])
    trial_metrics_mean = trial_metrics.mean(axis=0)
    trial_metrics_std = trial_metrics.std(axis=0)
    print(f'Experiment {experiment_id}\n'
          f'finished with metrics:\n'
          f'mean={ff(trial_metrics_mean)}\n'
          f' std={ff(trial_metrics_std)}')
    return exp_name


def visualize_run_results(molecules: Iterable[MolGraph],
                          objective: Objective,
                          history: OptHistory,
                          save_path: Path,
                          show: bool = False):
    save_path.mkdir(parents=True, exist_ok=True)

    # Plot pareto front (if multi-objective)
    if objective.is_multi_objective:
        visualise_pareto(history.archive_history[-1],
                         objectives_names=objective.metric_names[:2],
                         folder=str(save_path))

    # Plot diversity
    history.show.diversity_population(save_path=save_path / 'diversity.gif')
    history.show.diversity_line(save_path=save_path / 'diversity_line.png')

    # Plot found molecules
    rw_molecules = [mol.get_rw_molecule() for mol in set(molecules)]
    objectives = [objective.format_fitness(objective(mol)) for mol in set(molecules)]
    image = Draw.MolsToGridImage(rw_molecules,
                                 legends=objectives,
                                 molsPerRow=min(4, len(rw_molecules)),
                                 subImgSize=(1000, 1000),
                                 legendFontSize=50)
    image.save(save_path / 'best_molecules.png')
    if show:
        image.show()


def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-drug', type=str, default='CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    parser.add_argument('-data_paths', nargs='+',
                        default=os.path.join(project_root(), 'pipeline/result/GAN_all_valid.csv'), help='')
    parser.add_argument('-coformers_col_name', type=str, default='generated_coformers')
    parser.add_argument('-results_folder', type=str, default='GOLEM/results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parameter()
    drug = args.drug
    data_paths = args.data_paths
    col_name = args.coformers_col_name
    results_folder = args.results_folder
    if not isinstance(data_paths, list):
        data_paths = [data_paths]

    exp_ids = []
    for init_path in data_paths:
        exp_id = run_experiment(molecule_search_setup,
                                initial_data_path=init_path,
                                col_name=col_name,
                                max_heavy_atoms=50,
                                trial_timeout=60,
                                trial_iterations=200,
                                pop_size=200,
                                visualize=True,
                                num_trials=10,
                                drug=drug,
                                result_dir=results_folder,
                                adaptive_kind=MutationAgentTypeEnum.random,
                                pretrain_dir=None
                                )
        exp_ids.append(exp_id)
        exp_results_folder = os.path.join(results_folder, exp_id)

        init_df = pd.read_csv(init_path)
        golem_result = get_final_dataset(exp_results_folder, drug)
        visualize_results(['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 'sa_score'],
                          init_df,
                          golem_result,
                          exp_results_folder)
        get_statistical_significance(['unobstructed', 'orthogonal_planes', 'h_bond_bridging'],
                                     init_df,
                                     golem_result,
                                     exp_results_folder)
    plot_experiment_comparison(exp_ids, metric_ids=[0, 1, 2, 3], results_dir=results_folder)
