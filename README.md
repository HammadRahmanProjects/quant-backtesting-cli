# Backtesting CLI Application

A Python-based quantitative research and backtesting system for testing, optimizing, and validating trading strategies across different assets and market regimes.

## Features
- Portfolio creation and storage
- Data cleaning and validation pipeline
- Multiple trading strategies
- Backtesting and risk analysis
- Parameter optimization with multiprocessing
- Heatmaps and equity curve visualization

## Status
Work in progress.

- Current features being implemented:
    - Working on cleaning the entire codebase
    - Making the optimization function quicker
        - Want to be able to feasibly support 100,000+ combinations tests
        - In future potentially 1,000,000+
        - rewrite of some active layers in numpys 
        - Currently we are running 30,000 tests in roughly 40s 