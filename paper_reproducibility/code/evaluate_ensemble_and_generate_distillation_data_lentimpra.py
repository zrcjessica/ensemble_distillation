#!/usr/bin/env python3
import sys, runpy
sys.argv = ['evaluate_ensemble_and_generate_distillation_data.py', '--dataset', 'lentiMPRA'] + sys.argv[1:]
runpy.run_path('evaluate_ensemble_and_generate_distillation_data.py', run_name='__main__')
