# Thesis Draft Management

## Overview

This directory contains the complete thesis draft materials, organized by chapters, figures, and tables. The structure follows Amirkabir University of Technology (AUT) thesis formatting requirements.

## Thesis Structure

### Chapter Organization

#### `chapters/`
Individual thesis chapters in LaTeX format.

**Chapter 1: Introduction**
- `chapter1_introduction.tex`
- Background and motivation
- Problem statement
- Research questions
- Thesis objectives
- Expected contributions

**Chapter 2: Literature Review**
- `chapter2_literature_review.tex`
- Historical evolution of EPF
- Methodological analysis
- Research gaps identification
- Theoretical foundations
- Synthesis and positioning

**Chapter 3: Research Methodology**
- `chapter3_methodology.tex`
- Data collection procedures
- Preprocessing pipeline
- Feature engineering
- Model implementation
- Evaluation framework

**Chapter 4: Model Implementation**
- `chapter4_models.tex`
- Baseline models (Linear, Ridge, Lasso)
- Tree-based models (RF, GB, XGBoost)
- Other ML models (KNN, SVR, MLP)
- Deep learning (LSTM)
- Hyperparameter optimization

**Chapter 5: Experimental Results**
- `chapter5_results.tex`
- Model comparison analysis
- Multi-resolution performance
- Feature fusion impact
- Volatility analysis
- Computational efficiency

**Chapter 6: Discussion and Analysis**
- `chapter6_discussion.tex`
- Results interpretation
- Research questions answered
- Practical implications
- Limitations and constraints
- Future research directions

**Chapter 7: Conclusion**
- `chapter7_conclusion.tex`
- Summary of contributions
- Theoretical implications
- Practical applications
- Academic impact
- Final recommendations

### Supporting Materials

#### `figures/`
All figures and visualizations for the thesis.

**Categories**
- `methodology/`: Process flowcharts, architecture diagrams
- `data/`: Data visualizations, statistical plots
- `results/`: Performance charts, comparison graphs
- `analysis/`: SHAP plots, feature importance
- `appendix/`: Additional supporting figures

**File Naming Convention**
- `fig_chapter_number_description.extension`
- Example: `fig_3_1_data_pipeline.png`

**Quality Standards**
- Minimum 300 DPI resolution
- Consistent styling and formatting
- Caption and source attribution
- Vector format when possible

#### `tables/`
All tables and tabular data presentations.

**Categories**
- `literature/`: Literature review summary tables
- `data/`: Dataset statistics and characteristics
- `results/`: Performance comparison tables
- `analysis/`: Statistical analysis results
- `appendix/`: Additional supporting tables

**Formatting Standards**
- AUT thesis table formatting
- Consistent column alignment
- Proper unit notation
- Statistical significance indicators

### Main Thesis Files

#### `main.tex`
Main thesis compilation file.
```latex
\documentclass[12pt,a4paper]{report}
\input{preamble}
\begin{document}
\input{chapters/title_page}
\input{chapters/abstract}
\tableofcontents
\input{chapters/chapter1_introduction}
\input{chapters/chapter2_literature_review}
\input{chapters/chapter3_methodology}
\input{chapters/chapter4_models}
\input{chapters/chapter5_results}
\input{chapters/chapter6_discussion}
\input{chapters/chapter7_conclusion}
\bibliography{references}
\appendix
\input{appendices}
\end{document}
```

#### `preamble.tex`
Document configuration and package imports.
- Document class and formatting
- Required packages (math, graphics, tables)
- Custom commands and environments
- Page layout and styling

#### `references.bib`
Complete bibliography in BibTeX format.
- All cited literature
- Proper formatting for AUT requirements
- Consistent citation style

## Writing Guidelines

### Language Requirements
- **English**: Main thesis document
- **Persian**: Abstract (چکیده) - max 250 words
- **Keywords**: 3-5 keywords in both languages

### Formatting Standards
- **Font**: Times New Roman, 12pt
- **Spacing**: 1.5 line spacing
- **Margins**: AUT standard margins
- **Page Numbers**: Bottom center, Roman numerals for front matter

### Citation Style
- **In-text**: Numerical citations [1], [2,3]
- **Bibliography**: Numerical order, AUT format
- **References**: Complete and consistent formatting

## Quality Assurance

### Content Review
- Supervisor review at each chapter
- Peer review from research group
- Language editing for clarity
- Technical accuracy verification

### Formatting Check
- AUT template compliance
- Consistent figure/table numbering
- Proper citation formatting
- PDF generation testing

### Plagiarism Check
- Originality verification
- Proper citation practices
- Paraphrasing guidelines
- Reference completeness

## Submission Requirements

### Final Document
- Complete thesis PDF
- All source files (.tex, .bib, figures)
- Supplementary materials
- Declaration of originality

### Electronic Submission
- AUT thesis repository
- PDF with embedded fonts
- Proper metadata
- Access permissions

### Print Requirements
- Hard copy for department
- Binding specifications
- Quality paper requirements
- Signature pages

## Timeline Management

### Writing Schedule
- **Months 1-2**: Chapters 1-2 (Introduction, Literature)
- **Months 3-4**: Chapters 3-4 (Methodology, Models)
- **Months 5-6**: Chapters 5-6 (Results, Discussion)
- **Month 7**: Chapter 7 (Conclusion) and final review

### Milestone Reviews
- Chapter completion reviews
- Mid-thesis progress assessment
- Pre-submission review
- Final defense preparation

## Collaboration Tools

### Version Control
- Git repository for version tracking
- Branch management for drafts
- Commit message standards
- Backup procedures

### Document Sharing
- Cloud storage for drafts
- Collaborative editing platforms
- Comment and review tools
- Change tracking

## Resources and Templates

### AUT Templates
- Official thesis template
- Formatting guidelines
- Submission procedures
- Evaluation criteria

### LaTeX Resources
- Template customization
- Package documentation
- Troubleshooting guides
- Best practices

### Writing Support
- University writing center
- Technical writing resources
- English language support
- Citation management tools