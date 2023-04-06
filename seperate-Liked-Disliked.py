import pandas as pd
import shutil

def main():
    df = pd.read_csv(r'data/overall_ad_ratings.csv')
    for i in df.index:
        if df['overall_rating'][i] >= 4:
            source = "screenshots/" + str(df['ad_id'][i])  + ".webp"
            target = "liked/" + str(df['ad_id'][i]) + "@" + str(df['pid'][i]) + ".webp"
            shutil.copy(source, target)
        else:
            source = "screenshots/" + str(df['ad_id'][i])  + ".webp"
            target = "disliked/" + str(df['ad_id'][i]) + "@" + str(df['pid'][i]) + ".webp"
            shutil.copy(source, target)

main()