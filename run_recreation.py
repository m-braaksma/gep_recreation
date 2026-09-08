import os
import hazelbean as hb
import recreation_tasks

# Reset GDAL_DATA path after importing hazelbean
# Mac error using pygrio engine for geopandas read_file
# ValueError: Found GDAL data directory at * but it does not appear to correctly contain GDAL data files
gdal_data_path = os.environ.get("GDAL_DATA")
print("GDAL_DATA:", gdal_data_path)
os.environ['GDAL_DATA'] = '/Users/mbraaksma/mambaforge/envs/hb01/share/gdal'


def build_recreation_task_tree(p):
    """Build the task tree for recreation analysis"""
    # Shared tasks
    p.generate_environment_index_task = p.add_task(recreation_tasks.generate_environment_index)
    p.generate_accessibility_index_task = p.add_task(recreation_tasks.generate_accessibility_index)
    p.recreation_site_analysis_task = p.add_task(recreation_tasks.recreation_site_analysis)

    # Tourism Preprocessing
    p.allocate_overnights_task = p.add_task(recreation_tasks.allocate_overnights)

    # Daily Recreation
    p.calculate_daily_rec_task = p.add_task(recreation_tasks.calculate_daily_recreation_into_sites_kernel)

    # Tourism
    p.calculate_tourist_recreation_task = p.add_task(recreation_tasks.calculate_tourist_recreation_into_sites_kernel)

    # Post-processing and visualization
    p.aggregate_results_task = p.add_task(recreation_tasks.aggregate_results_iso3)

    return p


if __name__ == '__main__':
    hb.log('Starting GEP recreation workflow...')
    
    # Create the ProjectFlow object
    p = hb.ProjectFlow()
    p.force_run = False  # Set to True if you want to force rerun all tasks
    p.L = hb.get_logger('recreation_workflow')
    
    # Base directories
    p.user_dir = os.path.expanduser('~')
    p.data_dir = '/Users/mbraaksma/Library/CloudStorage/GoogleDrive-braak014@umn.edu/Shared drives/NatCapTEEMs/Projects/Global GEP/Ecosystem Services SubFolders/Recreation/data/'
    
    # Project setup
    p.extra_dirs = ['Files', 'gep_recreation', 'projects']  
    p.project_name = 'global_recreation_results'
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)
    
    # Log the configuration
    p.L.info(f"Data directory: {p.data_dir}")
    p.L.info(f"Project directory: {p.project_dir}")
    
    # Build task tree
    build_recreation_task_tree(p)
    
    # Execute the workflow
    p.execute()
