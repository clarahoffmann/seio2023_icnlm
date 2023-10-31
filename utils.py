
import pandas as pd

def read_matched_paths(home_dir):
  """
  reads in paths for all images and labels and match files
  and returns a df with matched paths

  home_dir: path of 'LiveEO_ML_Engineer_challenge' folder
  """
  
  # read paths of all files
  img_paths = glob.glob(home_dir + "images/*")
  lbs_pix_paths = glob.glob(home_dir + "labels_pix/*")
  lbs_geo_paths = glob.glob(home_dir + "labels_geo/*")
  
  # get identifier for all files
  # images
  img_paths_df = pd.DataFrame({'full_paths_img': img_paths})
  img_paths_df['id'] = [x.split("images/", 1)[1].split(".tif", 1)[0] for x in  img_paths_df.full_paths_img]

  # label in vector format
  lbs_pix_df = pd.DataFrame({'full_paths_lbs': lbs_pix_paths})
  lbs_pix_df['id'] = [x.split("labels_pix/", 1)[1].split("_Buildings.geojson", 1)[0] for x in  lbs_pix_df.full_paths_lbs]

  # labels in geojson format
  lbs_geo_df = pd.DataFrame({'full_paths_lbs': lbs_geo_paths})
  lbs_geo_df['id'] = [x.split("labels_geo/", 1)[1].split("_Buildings.geojson", 1)[0] for x in  lbs_geo_df.full_paths_lbs]

  # merge to one df 
  df_paths = img_paths_df.merge(lbs_pix_df, left_on='id', right_on='id').merge(lbs_geo_df, left_on='id', right_on='id').drop(['id'], axis=1)
  
  return(df_paths)