# List of all features
A list of all features and our (educated guess) of how important they are.


**'store_id'**: FK to other DFs. IMPORTANCE LEVEL: 0

**'year'**: not relevant, always 2016. IMPORTANCE LEVEL: 0

**'store_name'**: almost always unique, more relevant information in chain name, plaace_hierarchy_id and (lat, lon). IMPORTANCE LEVEL: 0

**'plaace_hierarchy_id'**: type of store. IMPORTANCE LEVEL: 5
<ul>
<li>'plaace_cat_{p}': plaace ID split into 4 levels of abstraction. IMPORTANCE LEVEL: 5</li>
<li>'is_grocery': whether or not something is a grocery store (as most outliers were grocery stores): IMPORTANCE LEVEL: 3 </li>
</ul>

**'sales_channel_name'**: holds same information as plaace_hierarchy_id. IMPORTANCE LEVEL: 1

**'grunnkrets_id'**: holds information about grunnkrets, delomrade, kommune and fylke. IMPORTANCE LEVEL: 5
<ul>
<li>'fylke', 'kommune', 'delomrade': derived from grunnkrets. IMPORTANCE LEVEL: 5</li>
</ul>

**'address'**: hard to use, location of shop can be extracted from (lat, lon) or grunnkrets_id. IMPORTANCE LEVEL: 0

**'lat'**: same information as grunnkrets_id, skip. IMPORTANCE LEVEL: 1
<ul>
<li>'point': (lat, lon). Used in other functions, but not important itself. IMPORTANCE LEVEL: 1</li>
</ul>

**'lon'**: same information as grunnkrets_id, skip. IMPORTANCE LEVEL: 1
<ul>
<li>'point': (lat, lon). Used in other functions, but not important itself. IMPORTANCE LEVEL: 1</li>
</ul>

**'chain_name'**: many NANs, need to be handled. IMPORTANCE LEVEL: 3
<ul>
<li>'is_chain': boolean, chain or not. IMPORTANCE LEVEL: 3</li>
<li>'bounded_chain_name': only include chains with certain size. Seems like only using chain name is in fact better. IMPORTANCE LEVEL: 2</li>
<li>'chain_mean_revenue': mean revenue of all other stores in chain. IMPORTANCE LEVEL: 4</li>
</ul>

**'mall_name'**: many NANs, seems incomplete. IMPORTANCE LEVEL: 2
<ul>
<li>'is_mall': boolean, mall or not. IMPORTANCE LEVEL: 3</li>
</ul>

**Distance to competitors:**
<ul>
<li>'sum_dist_to_nearest_{n}_comp_plaace_{p}': sum of distance to n nearest competitors with a matchin plaace_cat{p} (higher p -> finer categories). IMPORTANCE LEVEL: 3</li>
<li>'mean_dist_to_nearest_{n}_comp_plaace_{p}': mean of distance to n nearest competitors with a matchin plaace_cat{p} (higher p -> finer categories). IMPORTANCE LEVEL: 3</li>
<li>Maybe enough to use one of these?</li>
</ul>

**Distance to bus stops:**
<ul>
<li>'closest_bus_stop_sum_{n}': sum of distance to n nearest bus stops. Doesn't appear that correlated: IMPORTANCE LEVEL: 2</li>
<li>'closest_bus_stop_mean_{n}': mean of distance to n nearest bus stops. Doesn't appear that correlated: IMPORTANCE LEVEL: 2</li>
<li>Again, maybe enough to use one of these?</li>
</ul>

**Mean revenue of similar stores:**
<ul>
<li>'mean_revenue_{p}': mean revenue of all stores with similar plaace_cat_{p}. Seems quite correlated. IMPORTANCE LEVEL: 3</li>
'log_mean_revenue_{p}': log of mean revenue of all stores with similar plaace_cat_{p}. Seems more correlated, as we use log of response variable when training. IMPORTANCE LEVEL: 4</li>
</ul>

'{district}.age_{low}-{high}':

'{district}.c_age_{low}-{high}':

'{district}.mean_age':

'{district}.tot_pop':

'{district}.c_age_{low}-{high}_ratio':

'{district}.tot_pop_log':

'{district}.area_km2':

'{district}.couple_children_0_to_5_years':

'{district}.couple_children_18_or_above':

'{district}.couple_children_6_to_17_years':

'{district}.couple_without_children':

'{district}.single_parent_children_0_to_5_years':

'{district}.single_parent_children_18_or_above':

'{district}.single_parent_children_6_to_17_years':

'{district}.singles':

'{district}.tot_household':

'{district}.all_households_income':

'{district}.singles_income':

'{district}.couple_without_children_income':

'{district}.couple_with_children_income':

'{district}.other_households_income':

'{district}.single_parent_with_children_income':

'{district}.total_income':

'{district}.total_income_log':

'{district}.income_density':

'{district}.income_density_log':

'{district}.pop_density':

'{district}.pop_density_log':

**'revenue'**: response. IMPORTANCE LEVEL: N/A

**'log_revenue'**: log of response. Better to use this when training, then transform guesses back. IMPORTANCE LEVEL: N/A