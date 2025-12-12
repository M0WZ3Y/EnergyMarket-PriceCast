# Literature Review Management

## Overview

This directory contains all literature review materials for the electricity price forecasting thesis. The literature review is based on 37 curated papers covering the evolution from traditional ARIMA models to modern deep learning approaches.

## Directory Structure

### `papers/`
Collection of research papers in various formats.

**Organization**
- Papers organized by publication year
- Subdirectories by methodology category
- Full-text PDFs and supplementary materials
- Citation metadata and notes

**Categories**
- `traditional/`: ARIMA, statistical methods
- `machine_learning/`: Classical ML approaches
- `deep_learning/`: Neural network methods
- `hybrid/`: Combined methodologies
- `review/`: Survey and review papers

### `bibTeX/`
Bibliography management and citation database.

**Files**
- `references.bib`: Complete bibliography in BibTeX format
- `categorized/`: BibTeX files by research category
- `templates/`: Citation style templates
- `export/`: Formatted bibliographies for different journals

### `literature_summary.md`
Comprehensive literature review synthesis.

**Content Structure**
1. **Historical Evolution**
   - Early statistical methods (ARIMA, regression)
   - Machine learning revolution
   - Deep learning emergence
   - Current state-of-the-art

2. **Methodological Analysis**
   - Performance comparison studies
   - Feature engineering approaches
   - Model evaluation metrics
   - Validation methodologies

3. **Research Gaps Identification**
   - Unified model comparison needs
   - Explainability requirements
   - Feature fusion inconsistencies
   - Multi-resolution handling

4. **Theoretical Foundations**
   - Time series forecasting theory
   - Market dynamics modeling
   - Volatility modeling approaches
   - Risk assessment frameworks

## Key Research Findings

### Evolution Timeline
- **2005**: Conejo et al. - Wavelet transform + ARIMA
- **2018**: Lago et al. - Deep learning comprehensive comparison
- **2022**: Jedrzejewski et al. - ML dawn in EPF
- **2024**: Mubarak et al. - CNN-BiLSTM hybrid models
- **2025**: Ghelasi & Ziel - Multi-horizon econometric models

### Methodological Insights

**Traditional Methods**
- ARIMA models: Good baseline, limited nonlinearity handling
- Statistical approaches: Interpretable, struggle with complexity
- Regression models: Simple, require feature engineering

**Machine Learning**
- Ensemble methods (XGBoost, Random Forest): Excel in nonlinearity
- SVM: Effective in high-dimensional spaces
- KNN: Simple, sensitive to noise

**Deep Learning**
- LSTM: Superior temporal pattern capture
- CNN: Spatial feature extraction
- Hybrid models: Combine strengths of multiple approaches

### Identified Research Gaps

**Gap 1: Unified Model Comparison**
- Fragmented evaluations with inconsistent datasets
- Limited generalizability across studies
- Need for standardized benchmarking

**Gap 2: Model Explainability**
- "Black-box" nature of advanced models
- Limited transparency in decision-making
- Stakeholder trust issues

**Gap 3: Feature Fusion**
- Inconsistent use of exogenous variables
- Missing quantified impact assessments
- Lack of systematic feature integration

### Performance Benchmarks

**Accuracy Metrics**
- MAPE: Typically 5-15% for modern approaches
- RMSE: Varies by market volatility
- MAE: Consistent across methodologies

**Computational Considerations**
- Training time: Deep learning > ensemble > traditional
- Inference speed: Traditional > ensemble > deep learning
- Memory requirements: Deep learning significantly higher

## Citation Management

### BibTeX Structure
```bibtex
@article{conejo2005day,
  title={Day-ahead electricity price forecasting using the wavelet transform and ARIMA models},
  author={Conejo, Antonio J and Plazas, Miguel A and Espinola, Rosa and Molina, Ana B},
  journal={IEEE Transactions on Power Systems},
  volume={20},
  number={2},
  pages={1035--1042},
  year={2005}
}
```

### Citation Categories
1. **Methodological Papers**: Core algorithm development
2. **Comparison Studies**: Model performance analysis
3. **Application Papers**: Real-world implementations
4. **Review Papers**: Survey and synthesis work

## Quality Assessment

### Inclusion Criteria
- Peer-reviewed journal articles
- Conference proceedings with high impact
- Recent publications (last 10 years priority)
- Relevant to electricity price forecasting

### Exclusion Criteria
- Non-English publications (unless critical)
- Low-impact or predatory journals
- Irrelevant application domains
- Incomplete methodological descriptions

## Synthesis Framework

### Research Question Mapping
- **RQ1**: Model comparison studies
- **RQ2**: Multi-resolution analysis papers
- **RQ3**: Feature engineering research
- **RQ4**: Volatility modeling literature
- **RQ5**: Explainability and interpretability
- **RQ6**: Computational efficiency studies

### Contribution Analysis
- **Theoretical**: Novel methodologies and frameworks
- **Practical**: Real-world applications and implementations
- **Empirical**: Experimental results and validations
- **Review**: Synthesis and future directions

## Future Research Directions

### Emerging Trends
- Transformer architectures for time series
- Graph neural networks for market modeling
- Federated learning for privacy preservation
- Quantum computing applications

### Methodological Gaps
- Real-time forecasting capabilities
- Multi-market transfer learning
- Uncertainty quantification
- Causal inference integration

## Tools and Resources

### Reference Management
- **Paperpile**: Online reference manager
- **Zotero**: Open-source alternative
- **Mendeley**: Academic social network
- **EndNote**: Professional reference software

### Analysis Tools
- **VOSviewer**: Bibliometric analysis
- **CiteSpace**: Citation pattern analysis
- **Bibliometrix**: R package for bibliometrics
- **Connected Papers**: Visual exploration

## Collaboration and Sharing

### Research Network
- Supervisor and advisor guidance
- Peer review from research group
- Conference networking opportunities
- Online research communities

### Open Science Practices
- Preprint sharing (arXiv, SSRN)
- Code and data availability
- Reproducible research practices
- Open access publication