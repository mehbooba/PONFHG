
'''

CODE USED TO CONVERT THE DATASET TO ADD EXTRA FIELDS FROM OTHER FILES 

SO THAT FUZZIFICATION CAN BE DONE


'''

from sklearn.preprocessing import LabelEncoder

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
#nltk.download('vader_lexicon')
...
data=pd.read_csv('/home/mehbooba/Documents/IFHGRec/data/coco/evaluate.csv')
#average  rating for a course
data['course_avg_rating'] = data.groupby('course_id')['learner_rating'].transform('mean')
#total number of ratings of a course
data['Counts'] = data.groupby(['course_id'])['learner_rating'].transform('count')
tdata=pd.read_csv('/home/mehbooba/Documents/IFHGRec/data/coco/teach.csv')
insdata=pd.read_csv('/home/mehbooba/Documents/IFHGRec/data/coco/instructor.csv')
coursedata=pd.read_csv('/home/mehbooba/Documents/IFHGRec/data/coco/course.csv')

cols = ['course_id']
#to add instructor id
data=pd.merge(data, tdata, on="course_id",how='left')
#to add instructor details
data=pd.merge(data, insdata, on="instructor_id",how='left')

data['instructr_perf']=data['total_enrollments']/data['total_courses']

columns_to_drop = ['Unnamed: 0_x','learner_timestamp', 'review_id', 'Unnamed: 0_y', 'instructor_id', 'job_title', 'total_reviews', 'total_enrollments', 'total_courses']
data= data.drop(columns=columns_to_drop)
print(list(data))

#label encoding of categorical values from course

label_encoder = LabelEncoder()

# Apply label encoder to each column

coursedata['instructional_level_encoded'] = label_encoder.fit_transform(coursedata['instructional_level'])
coursedata['language_encoded'] = label_encoder.fit_transform(coursedata['language'])
coursedata['2nd_level_category_encoded'] = label_encoder.fit_transform(coursedata['second_level_category'])

cols_to_keep = ['course_id','instructional_level_encoded', 'language_encoded', '2nd_level_category_encoded','short_description']

# Keep only the specified columns
coursedata_filtered = coursedata.loc[:, cols_to_keep]

data=pd.merge(data, coursedata_filtered, on="course_id",how='left')


#normalisation of needed fields

data['n_course_avg_rating'] = data['course_avg_rating'] /data['course_avg_rating'].abs().max()
data['n_Counts'] = data['Counts'] /data['Counts'].abs().max()
data['n_instructr_perf'] = data['instructr_perf'] /data['instructr_perf'].abs().max()

data['learner_comment'] = data['learner_comment'].astype(str).fillna("")

data['learner_comment'] = data['learner_comment'].astype(str).fillna("")

# Define a function to convert sentiments to 0-1 scale using TextBlob
def convert_sentiment_to_score(comment):
    # Create a TextBlob object
    blob = TextBlob(comment)
    # Get the polarity score which ranges from -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity
    # Normalize polarity score from -1 to 1 to 0 to 1
    normalized_score = (polarity + 1) / 2
    return normalized_score

# Apply the function to the 'learner_comment' column
data['sentiment_score'] = data['learner_comment'].apply(convert_sentiment_to_score)



data=data.fillna(0)
#print(data.head(30))
data.to_csv("LRP_COCO_data_fuzzy.csv")

print(list(data))
