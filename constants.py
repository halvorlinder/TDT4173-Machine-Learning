levels = ['grunnkrets_id', 'delomrade', 'kommune', 'fylke']
levels_ext = ['grunnkrets_id', 'delomrade', 'kommune', 'fylke', 'country']
next_nevel = {'grunnkrets_id':'delomrade', 'delomrade':'kommune', 'kommune':'fylke'}
next_nevel_ext = {'grunnkrets_id':'delomrade', 'delomrade':'kommune', 'kommune':'fylke', 'fylke':'country'}
income_cols = ['all_households', 'singles', 'couple_without_children',	'couple_with_children', 'other_households',	'single_parent_with_children']
plaace_cols = ['plaace_cat_1',	'plaace_cat_2',	'plaace_cat_3',	'plaace_cat_4']
next_plaace_col = { 'plaace_cat_2':'plaace_cat_1','plaace_cat_3':'plaace_cat_2','plaace_cat_4':'plaace_cat_3'}