# import unittest
# from unittest.mock import Mock

# from utama_core.entities.game import Field
# from utama_core.run.refiners.position import PositionRefiner

# full_field = Field.full_field_bounds
# position_refiner = PositionRefiner(True, 6, 6, full_field)

# # Test that the least squares errors for all 3 parameters is below baseline.
# def test_filter_reduces_error():
#     position_refiner.refine()