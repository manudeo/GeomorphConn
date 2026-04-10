"""
tables.py
=========
RUSLE C-factor lookup tables for land-cover-based IC weighting.

The C-factor (cover-management factor) is one of six factors in the Revised
Universal Soil Loss Equation (RUSLE; Renard et al. 1997).  It expresses the
ratio of soil loss under a specific cover/management condition to the
corresponding loss from continuously tilled bare fallow.  Values are
dimensionless and range from ≈0 (dense perennial canopy) to ≈1 (bare fallow).

**Source hierarchy**

[R97]  Renard, K.G., Foster, G.R., Weesies, G.A., McCool, D.K., & Yoder,
       D.C. (1997). Predicting soil erosion by water: a guide to conservation
       planning with the RUSLE. USDA Agricultural Handbook 703.
       (primary authority for all RUSLE C-factor values)

[W78]  Wischmeier, W.H. & Smith, D.D. (1978). Predicting rainfall erosion
       losses. USDA Agricultural Handbook 537.
       (C-factor ranges for cover types not revisited in RUSLE)

[P15]  Panagos, P. et al. (2015). Estimating the soil erosion
       cover-management factor at the European scale. Land Use Policy, 48,
       38-50.  (Europe-specific RUSLE C refinements; used for CORINE table)

[B08]  Borselli, L., Cassi, P., & Torri, D. (2008). Prolegomena to sediment
       and flow connectivity in the landscape. Catena, 75(3), 268-277.
       (the original IC paper using land-cover-derived C-factors)

.. note::
   C-factor values vary substantially with crop type, tillage practice,
   residue management, and climate.  Values here are representative central
   estimates; users should review and adjust for their region — especially
   cropland classes (code 40, 211, 12) where inter-annual variability is high.
   Pass a custom ``dict[int, float]`` to ``LandCoverWeight`` to override any
   or all entries.
"""

# ---------------------------------------------------------------------------
# ESA WorldCover 10 m v200 (2021) / v100 (2020)
# Class codes: integer pixel values in the WorldCover GeoTIFF.
# ---------------------------------------------------------------------------
# Mapping rationale (code: description → C source):
#   10 Tree cover        closed-canopy forest            0.003 [R97 §9, Table 9-1]
#   20 Shrubland         open shrub, 30–70 % cover       0.050 [R97 §9]
#   30 Grassland         continuous grass                0.040 [R97 §9]
#   40 Cropland          mixed arable (broad)            0.200 [R97 Table 9-1]
#   50 Built-up          impervious                      0.003 [W78]
#   60 Bare / sparse     desert, rock, <15 % cover       0.600 [R97 §9]
#   70 Snow and ice      —                               0.000
#   80 Permanent water   —                               0.000
#   90 Herbaceous wetland emergent marsh                 0.025 [W78]
#   95 Mangroves         dense root mat                  0.002 [R97 §9]
#  100 Moss and lichen   tundra / alpine mat             0.040 [W78]
#   -1 Nodata / unknown  conservative fallback           0.200
WORLDCOVER_C_FACTOR: dict[int, float] = {
    10: 0.003,
    20: 0.050,
    30: 0.040,
    40: 0.200,
    50: 0.003,
    60: 0.600,
    70: 0.000,
    80: 0.000,
    90: 0.025,
    95: 0.002,
    100: 0.040,
    -1: 0.200,
}

# ---------------------------------------------------------------------------
# CORINE Land Cover 2018, Level 3 (44 classes)
# ---------------------------------------------------------------------------
CORINE_C_FACTOR: dict[int, float] = {
    # Artificial surfaces
    111: 0.000,  # Continuous urban fabric           [W78]
    112: 0.003,  # Discontinuous urban fabric        [P15]
    121: 0.000,  # Industrial / commercial units     [W78]
    122: 0.000,  # Road and rail networks            [W78]
    123: 0.000,  # Port areas                        [W78]
    124: 0.000,  # Airports                          [W78]
    131: 0.350,  # Mineral extraction sites          [R97 §9]
    132: 0.450,  # Dump sites                        [R97 §9]
    133: 0.400,  # Construction sites                [R97 §9]
    141: 0.020,  # Green urban areas                 [P15]
    142: 0.025,  # Sport and leisure facilities      [P15]
    # Agricultural areas
    211: 0.220,  # Non-irrigated arable land         [R97 Table 9-1]
    212: 0.170,  # Permanently irrigated land        [R97 Table 9-1]
    213: 0.090,  # Rice fields                       [R97 Table 9-1]
    221: 0.100,  # Vineyards                         [P15]
    222: 0.080,  # Fruit trees and berry plantations [P15]
    223: 0.070,  # Olive groves                      [P15]
    231: 0.060,  # Pastures                          [R97 §9]
    241: 0.250,  # Annual crops with permanent crops [R97 Table 9-1]
    242: 0.210,  # Complex cultivation patterns      [R97 Table 9-1]
    243: 0.140,  # Agriculture with natural veg.     [P15]
    244: 0.090,  # Agro-forestry areas               [P15]
    # Forest and semi-natural areas
    311: 0.003,  # Broad-leaved forest               [R97 §9, Table 9-1]
    312: 0.004,  # Coniferous forest                 [R97 §9, Table 9-1]
    313: 0.003,  # Mixed forest                      [R97 §9]
    321: 0.040,  # Natural grasslands                [R97 §9]
    322: 0.030,  # Moors and heathland               [P15]
    323: 0.035,  # Sclerophyllous vegetation         [P15]
    324: 0.020,  # Transitional woodland-shrub       [P15]
    331: 0.500,  # Beaches, dunes, sands             [R97 §9]
    332: 0.700,  # Bare rocks                        [R97 §9]
    333: 0.400,  # Sparsely vegetated areas          [R97 §9]
    334: 0.450,  # Burnt areas (immediate post-fire) [R97 §9]
    335: 0.000,  # Glaciers and perpetual snow
    # Wetlands
    411: 0.020,  # Inland marshes                    [W78]
    412: 0.010,  # Peat bogs                         [W78]
    421: 0.020,  # Salt marshes                      [W78]
    422: 0.010,  # Salines                           [W78]
    423: 0.000,  # Intertidal flats
    # Water bodies
    511: 0.000,
    512: 0.000,
    521: 0.000,
    522: 0.000,
    523: 0.000,
    # Nodata
    -1: 0.200,
}

# ---------------------------------------------------------------------------
# MODIS MCD12Q1 Land Cover Type 1 (IGBP, 17 classes, 500 m)
# ---------------------------------------------------------------------------
MODIS_IGBP_C_FACTOR: dict[int, float] = {
    1: 0.003,   # Evergreen Needleleaf Forests  [R97 §9]
    2: 0.003,   # Evergreen Broadleaf Forests   [R97 §9]
    3: 0.004,   # Deciduous Needleleaf Forests  [R97 §9]
    4: 0.004,   # Deciduous Broadleaf Forests   [R97 §9]
    5: 0.003,   # Mixed Forests                 [R97 §9]
    6: 0.045,   # Closed Shrublands             [R97 §9]
    7: 0.070,   # Open Shrublands               [R97 §9]
    8: 0.055,   # Woody Savannas                [R97 §9]
    9: 0.075,   # Savannas                      [R97 §9]
    10: 0.040,  # Grasslands                    [R97 §9]
    11: 0.020,  # Permanent Wetlands            [W78]
    12: 0.200,  # Croplands                     [R97 Table 9-1]
    13: 0.003,  # Urban and Built-up Lands      [W78]
    14: 0.150,  # Cropland / Natural Veg. Mosaic[R97 Table 9-1]
    15: 0.000,  # Snow and Ice
    16: 0.600,  # Barren                        [R97 §9]
    17: 0.000,  # Water Bodies
    -1: 0.200,  # Nodata (fallback)
}

# Convenience alias
DEFAULT_C_FACTOR_TABLE = WORLDCOVER_C_FACTOR
