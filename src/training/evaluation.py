def evaluate_model(model, test_data, metrics):
    """
    Módulo de Avaliação Avançada para JurisOracle
    ===========================================

    Este módulo fornece métricas e ferramentas de avaliação especializadas
    para modelos de processamento de linguagem natural no domínio jurídico.

    Autor: JurisOracle Team
    Data: 2024
    Versão: 3.0.0
    """
    pass

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import statistics

# ML/NLP Libraries
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Text Processing
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

# Specialized Libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# Internal Imports
try:
    from ..core.logging_config import setup_logging
except ImportError:
    # Fallback: define a dummy logger if import fails
    def setup_logging(name):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
from ..models.base import BaseModel

# Helper to get config (dummy fallback)
def get_config():
    return {}

# Setup Logging
logger = setup_logging(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class EvaluationMetrics:
    """Classe para armazenar métricas de avaliação."""
    
    # Basic Metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Text Generation Metrics
    bleu_score: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    bert_score: Optional[float] = None
    
    # Retrieval Metrics
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    ndcg: Optional[float] = None  # Normalized Discounted Cumulative Gain
    hit_rate: Optional[float] = None
    
    # Legal-Specific Metrics
    legal_accuracy: Optional[float] = None
    citation_accuracy: Optional[float] = None
    jurisprudence_relevance: Optional[float] = None
    
    # Statistical Metrics
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    
    # Additional Info
    sample_size: Optional[int] = None
    evaluation_time: Optional[float] = None
    model_version: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Converte para JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

class MetricsCalculator:
    """Calculadora de métricas avançadas para domínio jurídico."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smoothing_function = SmoothingFunction().method1
        self._tfidf_cache = {}
        logger.info("MetricsCalculator inicializado com sucesso")
    # ...existing code for MetricsCalculator methods...

class ModelEvaluator:
    """Avaliador principal de modelos para JurisOracle."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.metrics_calculator = MetricsCalculator(config)
        self.tracking_enabled = False
        if WANDB_AVAILABLE and self.config.get('wandb', {}).get('enabled', False):
            self.setup_wandb()
        elif MLFLOW_AVAILABLE and self.config.get('mlflow', {}).get('enabled', False):
            self.setup_mlflow()
        self.evaluation_history = []
        logger.info("ModelEvaluator inicializado com sucesso")
    # ...existing code for ModelEvaluator methods...

class BenchmarkSuite:
    """Suite de benchmarks para modelos jurídicos."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.evaluator = ModelEvaluator(config)
        self.available_benchmarks = {
            'legal_qa': 'Benchmark de Question Answering Jurídico',
            'case_summarization': 'Benchmark de Sumarização de Casos',
            'legal_classification': 'Benchmark de Classificação Jurídica',
            'citation_extraction': 'Benchmark de Extração de Citações',
            'jurisprudence_search': 'Benchmark de Busca de Jurisprudência'
        }
        logger.info("BenchmarkSuite inicializada com sucesso")
    def run_benchmark(self, benchmark_name: str, model: Any, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            if benchmark_name not in self.available_benchmarks:
                raise ValueError(f"Benchmark '{benchmark_name}' não disponível")
            logger.info(f"Executando benchmark: {benchmark_name}")
            if dataset_path:
                test_data = self._load_custom_dataset(dataset_path)
            else:
                test_data = self._load_default_dataset(benchmark_name)
            if benchmark_name == 'legal_qa':
                results = self.evaluator.evaluate_qa_model(model, test_data)
            elif benchmark_name == 'case_summarization':
                results = self.evaluator.evaluate_summarization_model(model, test_data)
            else:
                results = self._run_custom_benchmark(benchmark_name, model, test_data)
            benchmark_info = {
                'benchmark_name': benchmark_name,
                'benchmark_description': self.available_benchmarks[benchmark_name],
                'dataset_size': len(test_data),
                'evaluation_results': results,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Benchmark {benchmark_name} concluído com sucesso")
            return benchmark_info
        except Exception as e:
            logger.error(f"Erro ao executar benchmark {benchmark_name}: {e}")
            raise
    def run_full_evaluation(self, model: Any, benchmarks: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            if benchmarks is None:
                benchmarks = list(self.available_benchmarks.keys())
            logger.info(f"Iniciando avaliação completa com {len(benchmarks)} benchmarks")
            results = {
                'model_info': {
                    'model_type': type(model).__name__,
                    'evaluation_timestamp': datetime.now().isoformat()
                },
                'benchmark_results': {},
                'summary': {}
            }
            for benchmark_name in benchmarks:
                try:
                    benchmark_result = self.run_benchmark(benchmark_name, model)
                    results['benchmark_results'][benchmark_name] = benchmark_result
                except Exception as e:
                    logger.error(f"Erro no benchmark {benchmark_name}: {e}")
                    results['benchmark_results'][benchmark_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            results['summary'] = self._generate_evaluation_summary(results['benchmark_results'])
            logger.info("Avaliação completa concluída")
            return results
        except Exception as e:
            logger.error(f"Erro na avaliação completa: {e}")
            raise
    def _load_default_dataset(self, benchmark_name: str) -> List[Dict]:
        if benchmark_name == 'legal_qa':
            return [{
                'question': 'Qual a pena para furto?',
                'context': 'Código Penal Brasileiro...',
                'answer': 'Reclusão de 1 a 4 anos...'
            }]
        elif benchmark_name == 'case_summarization':
            return [{
                'document': 'Processo judicial completo...',
                'summary': 'Resumo do caso...'
            }]
        else:
            return []
    def _load_custom_dataset(self, dataset_path: str) -> List[Dict]:
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            if isinstance(dataset, list):
                return dataset
            elif isinstance(dataset, dict) and 'data' in dataset:
                return dataset['data']
            else:
                raise ValueError("Formato de dataset não reconhecido")
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {e}")
            raise
    def _run_custom_benchmark(self, benchmark_name: str, model: Any, test_data: List[Dict]) -> EvaluationMetrics:
        logger.warning(f"Benchmark customizado {benchmark_name} não implementado")
        return EvaluationMetrics()
    def _generate_evaluation_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            'total_benchmarks': len(benchmark_results),
            'successful_benchmarks': 0,
            'failed_benchmarks': 0,
            'average_scores': {},
            'best_performing_benchmark': None,
            'recommendations': []
        }
        scores_by_metric = defaultdict(list)
        for benchmark_name, result in benchmark_results.items():
            if 'error' in result:
                summary['failed_benchmarks'] += 1
            else:
                summary['successful_benchmarks'] += 1
                if 'evaluation_results' in result:
                    metrics = result['evaluation_results']
                    for attr_name in dir(metrics):
                        if not attr_name.startswith('_'):
                            value = getattr(metrics, attr_name)
                            if isinstance(value, (int, float)):
                                scores_by_metric[attr_name].append(value)
        for metric, scores in scores_by_metric.items():
            if scores:
                summary['average_scores'][metric] = np.mean(scores)
        if summary['average_scores']:
            best_metric = max(summary['average_scores'], key=summary['average_scores'].get)
            summary['best_performing_benchmark'] = best_metric
            if summary['average_scores'].get('legal_accuracy', 0) < 0.7:
                summary['recommendations'].append(
                    "Considere fine-tuning adicional para melhorar precisão jurídica"
                )
            if summary['average_scores'].get('f1_score', 0) < 0.6:
                summary['recommendations'].append(
                    "Modelo pode se beneficiar de mais dados de treinamento"
                )
        return summary

def evaluate_model(
    model: Any, 
    test_data: List[Dict], 
    model_type: str = 'qa',
    config: Optional[Dict] = None
) -> EvaluationMetrics:
    evaluator = ModelEvaluator(config)
    if model_type == 'qa':
        return evaluator.evaluate_qa_model(model, test_data)
    elif model_type == 'summarization':
        return evaluator.evaluate_summarization_model(model, test_data)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")

def run_benchmark_suite(
    model: Any, 
    benchmarks: Optional[List[str]] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    suite = BenchmarkSuite(config)
    return suite.run_full_evaluation(model, benchmarks)

if __name__ == "__main__":
    logger.info("Módulo de avaliação JurisOracle carregado com sucesso! 🏛️⚖️")
    try:
        calculator = MetricsCalculator()
        evaluator = ModelEvaluator()
        ref_text = "Este é um exemplo de texto jurídico sobre direito penal."
        pred_text = "Este é um exemplo sobre direito penal no sistema jurídico."
        bleu = calculator.calculate_bleu_score(ref_text, pred_text)
        rouge = calculator.calculate_rouge_scores(ref_text, pred_text)
        logger.info(f"BLEU Score: {bleu:.4f}")
        logger.info(f"ROUGE Scores: {rouge}")
        print("\n🎯 Sistema de Avaliação JurisOracle")
        print("=" * 50)
        print("✅ Componentes carregados com sucesso!")
        print(f"📊 BLEU Score de exemplo: {bleu:.4f}")
        print(f"📈 ROUGE-1 de exemplo: {rouge['rouge1']:.4f}")
        print("\n🚀 Sistema pronto para avaliação de modelos!")
    except Exception as e:
        logger.error(f"Erro na demonstração: {e}")
        print(f"❌ Erro na inicialização: {e}")
    def evaluate_summarization_model(
        self, 
        model: Any,
        test_data: List[Dict],
        metrics: List[str] = None
    ) -> EvaluationMetrics:
        """
        Avalia modelo de sumarização.
        
        Args:
            model: Modelo a ser avaliado
            test_data: Dados de teste
            metrics: Métricas a calcular
            
        Returns:
            Métricas de avaliação
        """
        start_time = datetime.now()
        try:
            if metrics is None:
                metrics = ['rouge', 'bert_score', 'legal_accuracy']
            predictions = []
            references = []
            logger.info(f"Avaliando modelo de sumarização com {len(test_data)} amostras")
            for i, sample in enumerate(test_data):
                try:
                    document = sample['document']
                    expected_summary = sample['summary']
                    prediction = model.summarize(document)
                    predictions.append(prediction)
                    references.append(expected_summary)
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processadas {i + 1}/{len(test_data)} amostras")
                except Exception as e:
                    logger.error(f"Erro ao processar amostra {i}: {e}")
                    predictions.append("")
                    references.append(sample.get('summary', ''))
            evaluation_metrics = EvaluationMetrics()
            if 'rouge' in metrics:
                rouge_scores = [
                    self.metrics_calculator.calculate_rouge_scores(ref, pred)
                    for ref, pred in zip(references, predictions)
                ]
                evaluation_metrics.rouge_1 = np.mean([s['rouge1'] for s in rouge_scores])
                evaluation_metrics.rouge_2 = np.mean([s['rouge2'] for s in rouge_scores])
                evaluation_metrics.rouge_l = np.mean([s['rougeL'] for s in rouge_scores])
            if 'bert_score' in metrics:
                bert_scores = self.metrics_calculator.calculate_bert_score(references, predictions)
                evaluation_metrics.bert_score = bert_scores['f1']
            if 'legal_accuracy' in metrics:
                legal_metrics = self.metrics_calculator.calculate_legal_accuracy(predictions, references)
                evaluation_metrics.legal_accuracy = legal_metrics['legal_accuracy']
            avg_compression_ratio = np.mean([
                len(pred.split()) / len(ref.split()) if len(ref.split()) > 0 else 0
                for pred, ref in zip(predictions, references)
            ])
            evaluation_metrics.sample_size = len(test_data)
            evaluation_metrics.evaluation_time = (datetime.now() - start_time).total_seconds()
            evaluation_metrics.timestamp = datetime.now().isoformat()
            self._log_metrics(evaluation_metrics, 'summarization_model')
            self.evaluation_history.append({
                'model_type': 'summarization',
                'timestamp': evaluation_metrics.timestamp,
                'metrics': evaluation_metrics.to_dict(),
                'compression_ratio': avg_compression_ratio
            })
            logger.info(f"Avaliação de sumarização concluída em {evaluation_metrics.evaluation_time:.2f}s")
            return evaluation_metrics
        except Exception as e:
            logger.error(f"Erro na avaliação do modelo de sumarização: {e}")
            raise

    def evaluate_retrieval_model(
        self, 
        model: Any,
        test_queries: List[str],
        relevant_docs: List[List[int]],
        k_values: List[int] = None
    ) -> Dict[str, EvaluationMetrics]:
        start_time = datetime.now()
        try:
            if k_values is None:
                k_values = [1, 3, 5, 10, 20]
            results = {}
            logger.info(f"Avaliando modelo de recuperação com {len(test_queries)} consultas")
            for k in k_values:
                all_metrics = []
                for i, (query, relevant) in enumerate(zip(test_queries, relevant_docs)):
                    try:
                        retrieved = model.retrieve(query, k=k)
                        retrieved_ids = [doc.id for doc in retrieved]
                        metrics = self.metrics_calculator.calculate_retrieval_metrics(relevant, retrieved_ids, k)
                        all_metrics.append(metrics)
                    except Exception as e:
                        logger.error(f"Erro ao processar consulta {i}: {e}")
                        all_metrics.append({
                            'precision_k': 0.0, 'recall_k': 0.0, 'f1_k': 0.0,
                            'mrr': 0.0, 'hit_rate': 0.0, 'ndcg': 0.0
                        })
                evaluation_metrics = EvaluationMetrics()
                evaluation_metrics.precision = np.mean([m['precision_k'] for m in all_metrics])
                evaluation_metrics.recall = np.mean([m['recall_k'] for m in all_metrics])
                evaluation_metrics.f1_score = np.mean([m['f1_k'] for m in all_metrics])
                evaluation_metrics.mrr = np.mean([m['mrr'] for m in all_metrics])
                evaluation_metrics.hit_rate = np.mean([m['hit_rate'] for m in all_metrics])
                evaluation_metrics.ndcg = np.mean([m['ndcg'] for m in all_metrics])
                evaluation_metrics.sample_size = len(test_queries)
                evaluation_metrics.evaluation_time = (datetime.now() - start_time).total_seconds()
                evaluation_metrics.timestamp = datetime.now().isoformat()
                results[f'k_{k}'] = evaluation_metrics
                self._log_metrics(evaluation_metrics, f'retrieval_model_k_{k}')
            logger.info(f"Avaliação de recuperação concluída em {(datetime.now() - start_time).total_seconds():.2f}s")
            return results
        except Exception as e:
            logger.error(f"Erro na avaliação do modelo de recuperação: {e}")
            raise

    def compare_models(
        self, 
        model_results: Dict[str, EvaluationMetrics],
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        try:
            comparison_result = {
                'best_model': None,
                'rankings': {},
                'statistical_significance': {},
                'summary': {}
            }
            metrics_to_compare = ['f1_score', 'bleu_score', 'rouge_1', 'legal_accuracy']
            for metric in metrics_to_compare:
                metric_values = {}
                for model_name, metrics in model_results.items():
                    value = getattr(metrics, metric)
                    if value is not None:
                        metric_values[model_name] = value
                if metric_values:
                    best_model = max(metric_values, key=metric_values.get)
                    comparison_result['rankings'][metric] = sorted(
                        metric_values.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    if comparison_result['best_model'] is None:
                        comparison_result['best_model'] = best_model
            comparison_result['summary'] = {
                'total_models': len(model_results),
                'comparison_timestamp': datetime.now().isoformat(),
                'significance_level': significance_level
            }
            logger.info(f"Comparação de {len(model_results)} modelos concluída")
            return comparison_result
        except Exception as e:
            logger.error(f"Erro na comparação de modelos: {e}")
            raise

    def generate_evaluation_report(
        self, 
        evaluation_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"evaluation_report_{timestamp}.html"
            html_content = self._generate_html_report(evaluation_results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Relatório de avaliação salvo em: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            raise

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>JurisOracle - Relatório de Avaliação</title>
            <meta charset=\"utf-8\">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #1e3a8a; color: white; padding: 20px; border-radius: 8px; }
                .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; margin: 10px 0; border-radius: 8px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #1e40af; }
                .table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                .table th, .table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                .table th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class=\"header\">
                <h1>🏛️ JurisOracle - Relatório de Avaliação</h1>
                <p>Relatório gerado em: {timestamp}</p>
            </div>
        """.format(timestamp=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        if 'metrics' in results:
            html += "<h2>📊 Métricas Principais</h2>"
            metrics = results['metrics']
            for metric_name, value in metrics.to_dict().items():
                if value is not None and isinstance(value, (int, float)):
                    html += f"""
                    <div class=\"metric-card\">
                        <strong>{metric_name.replace('_', ' ').title()}:</strong>
                        <span class=\"metric-value\">{value:.4f}</span>
                    </div>
                    """
        if 'comparison' in results:
            html += "<h2>🔍 Comparação de Modelos</h2>"
        html += """
        </body>
        </html>
        """
        return html

    def _calculate_confidence_interval(
        self, 
        values: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        try:
            if len(values) < 2:
                return (0.0, 0.0)
            mean = np.mean(values)
            std_err = stats.sem(values)
            h = std_err * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
            return (mean - h, mean + h)
        except Exception as e:
            logger.error(f"Erro ao calcular intervalo de confiança: {e}")
            return (0.0, 0.0)

    def _log_metrics(self, metrics: EvaluationMetrics, model_type: str):
        try:
            if not self.tracking_enabled:
                return
            metrics_dict = metrics.to_dict()
            clean_metrics = {k: v for k, v in metrics_dict.items() if v is not None}
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({f"{model_type}_{k}": v for k, v in clean_metrics.items()})
            if MLFLOW_AVAILABLE:
                for key, value in clean_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{model_type}_{key}", value)
        except Exception as e:
            logger.error(f"Erro ao fazer log das métricas: {e}")

    def save_evaluation_history(self, filepath: str):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_history, f, indent=2, default=str)
            logger.info(f"Histórico de avaliações salvo em: {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")
            raise

    def load_evaluation_history(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.evaluation_history = json.load(f)
            logger.info(f"Histórico de avaliações carregado de: {filepath}")
        except Exception as e:
            logger.error(f"Erro ao carregar histórico: {e}")
            raise
    # ...existing code for BenchmarkSuite methods...
    # ...existing code...
    """Avaliador principal de modelos para JurisOracle."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o avaliador de modelos.
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or get_config()
        self.metrics_calculator = MetricsCalculator(config)
        
        # Configurar tracking de experimentos
        self.tracking_enabled = False
        if WANDB_AVAILABLE and self.config.get('wandb', {}).get('enabled', False):
            self.setup_wandb()
        elif MLFLOW_AVAILABLE and self.config.get('mlflow', {}).get('enabled', False):
            self.setup_mlflow()
        
        # Histórico de avaliações
        self.evaluation_history = []
        
        logger.info("ModelEvaluator inicializado com sucesso")
    
    def setup_wandb(self):
        """Configura Weights & Biases."""
        try:
            wandb_config = self.config.get('wandb', {})
            wandb.init(
                project=wandb_config.get('project', 'juris-oracle'),
                entity=wandb_config.get('entity'),
                config=self.config
            )
            self.tracking_enabled = True
            logger.info("Weights & Biases configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar Weights & Biases: {e}")
    
    def setup_mlflow(self):
        """Configura MLflow."""
        try:
            mlflow_config = self.config.get('mlflow', {})
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./mlruns'))
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'juris-oracle'))
            self.tracking_enabled = True
            logger.info("MLflow configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar MLflow: {e}")
    
    def evaluate_qa_model(
        self, 
        model: Any,
        test_data: List[Dict],
        metrics: List[str] = None
    ) -> EvaluationMetrics:
        """
        Avalia modelo de Question Answering.
        
        Args:
            model: Modelo a ser avaliado
            test_data: Dados de teste
            metrics: Métricas a calcular
            
        Returns:
            Métricas de avaliação
        """
        start_time = datetime.now()
        
        try:
            if metrics is None:
                metrics = ['bleu', 'rouge', 'bert_score', 'legal_accuracy']
            
            predictions = []
            references = []
            
            logger.info(f"Avaliando modelo QA com {len(test_data)} amostras")
            
            # Gerar predições
            for i, sample in enumerate(test_data):
                try:
                    question = sample['question']
                    context = sample.get('context', '')
                    expected_answer = sample['answer']
                    
                    # Fazer predição
                    prediction = model.predict(question, context)
                    
                    predictions.append(prediction)
                    references.append(expected_answer)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processadas {i + 1}/{len(test_data)} amostras")
                        
                except Exception as e:
                    logger.error(f"Erro ao processar amostra {i}: {e}")
                    predictions.append("")
                    references.append(sample.get('answer', ''))
            
            # Calcular métricas
            evaluation_metrics = EvaluationMetrics()
            
            if 'bleu' in metrics:
                bleu_scores = [
                    self.metrics_calculator.calculate_bleu_score(ref, pred)
                    for ref, pred in zip(references, predictions)
                ]
                evaluation_metrics.bleu_score = np.mean(bleu_scores)
            
            if 'rouge' in metrics:
                rouge_scores = [
                    self.metrics_calculator.calculate_rouge_scores(ref, pred)
                    for ref, pred in zip(references, predictions)
                ]
                evaluation_metrics.rouge_1 = np.mean([s['rouge1'] for s in rouge_scores])
                evaluation_metrics.rouge_2 = np.mean([s['rouge2'] for s in rouge_scores])
                evaluation_metrics.rouge_l = np.mean([s['rougeL'] for s in rouge_scores])
            
            if 'bert_score' in metrics:
                bert_scores = self.metrics_calculator.calculate_bert_score(references, predictions)
                evaluation_metrics.bert_score = bert_scores['f1']
            
            if 'legal_accuracy' in metrics:
                legal_metrics = self.metrics_calculator.calculate_legal_accuracy(
                    predictions, references
                )
                evaluation_metrics.legal_accuracy = legal_metrics['legal_accuracy']
                evaluation_metrics.citation_accuracy = legal_metrics['citation_accuracy']
            
            # Métricas adicionais
            evaluation_metrics.sample_size = len(test_data)
            evaluation_metrics.evaluation_time = (datetime.now() - start_time).total_seconds()
            evaluation_metrics.timestamp = datetime.now().isoformat()
            
            # Calcular intervalo de confiança para BLEU
            if evaluation_metrics.bleu_score is not None:
                bleu_scores = [
                    self.metrics_calculator.calculate_bleu_score(ref, pred)
                    for ref, pred in zip(references, predictions)
                ]
                confidence_interval = self._calculate_confidence_interval(bleu_scores)
                evaluation_metrics.confidence_interval = confidence_interval
            
            # Log das métricas
            self._log_metrics(evaluation_metrics, 'qa_model')
            
            # Salvar histórico
            self.evaluation_history.append({
                'model_type': 'qa',
                'timestamp': evaluation_metrics.timestamp,
                'metrics': evaluation_metrics.to_dict()
            })
            
            logger.info(f"Avaliação QA concluída em {evaluation_metrics.evaluation_time:.2f}s")
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação do modelo QA: {e}")
            raise
    def setup_wandb(self):
        """Configura Weights & Biases."""
        try:
            wandb_config = self.config.get('wandb', {})
            wandb.init(
                project=wandb_config.get('project', 'juris-oracle'),
                entity=wandb_config.get('entity'),
                config=self.config
            )
            self.tracking_enabled = True
            logger.info("Weights & Biases configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar Weights & Biases: {e}")

    def setup_mlflow(self):
        """Configura MLflow."""
        try:
            mlflow_config = self.config.get('mlflow', {})
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./mlruns'))
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'juris-oracle'))
            self.tracking_enabled = True
            logger.info("MLflow configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar MLflow: {e}")

    # ...existing code for MetricsCalculator methods...