"""Lowering implementation with explicit dependency directions.

The parent cuda_lowering.py is the compatibility facade. Mathematical functions
retain their bodies; dependencies point from dispatch/drivers to reusable
algebra, emission, policy and process helpers.
"""
