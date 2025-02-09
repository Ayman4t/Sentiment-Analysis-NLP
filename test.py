
from nlpProject import * 

#Preprocess the new text data
new_text = preprocess_text('''the american action film has been slowly drowning to death in a sea of asian wire-fu copycats . 
it's not a pretty death , and it's leaving the likes of schwartznager , stallone , and van damme wearing cement galoshes at the bottom of a kung fu sea . 
sometimes , the mix results in a mind-blowing spectacle unlike any other . 
quality action with amazing and exciting stunt work , as in 1999's the matrix , can be a real gem . 
but too often hollywood gets it wrong , even when they pay off chinese directors . 
flying ninjas and floating karate masters have been replaced by soaring bronx detectives and slow motion kicking scientists . 
mostly it's laughable . 
in hollywood's rush to emulate the success of the matrix , trademark asian stunt choreography has become more of a joke than an art form . 
but iron monkey , the latest asian import , shows us how to get it right . 
iron monkey ( actually a reissue of a 1993 film ) is the story of a 19th chinese vigilante ( rongguang yu ) , fighting with his own unique style of shaolin kung fu for the rights of the oppressed and the bellies of the hungry . 
but it is also a piece of the narrative of legendary chinese film hero wong fei-hong , most recently seen in one of the most overlooked , and possibly best films of 2000 , drunken master 2 ( released in the u . s . as the legend of drunken master ) . 
unlike drunken master 2 , which stars jackie chan as an adult fei-hong , iron monkey finds a much younger fei-hong ( sze-man tsang ) and his father wong kei-ying ( yen chi dan ) thrust into the middle of iron monkey's fight against oppression . 
iron monkey succeeds as no kung fu film since drunken master 2 . at times , fighting styles , especially that of monkey himself , do devolve into the ridiculous twinkle-toed floating of films like crouching tiger , hidden dragon , director yuen wo ping eventually remembers to bring his action scenes back to earth . 
iron monkey is at its heart a hardcore , kung fu action film rather than any kind of drama a la crouching tiger . 
however , there are brief moments of profoundness shared between characters , such as those that pass between our outlaw hero and his good-hearted but misguided enemy , chief fox . 
in those moments , and in others , iron monkey manages to transcend its mindless kung fu nature to touch the hearts and minds of its audience . 
while in no way the equal of a masterpiece like drunken master 2 , iron monkey dances quite nicely to the invading kung fu tune . 
aka siunin wong fei-hung tsi titmalau . 
''')

# Convert the preprocessed text to TF-IDF features using the same vectorizer
new_text_tfidf = vectorizer.transform([new_text])

# Make predictions using the trained models
# For example, with Logistic Regression
logistic_regression = joblib.load('logistic_regression_model.pkl')
logistic_pred = logistic_regression.predict(new_text_tfidf)

# Similarly, you can load and use other trained models
# For example, with Support Vector Machine
svm_classifier = joblib.load('Support_Vector_classifier_model.pkl')
svm_pred = svm_classifier.predict(new_text_tfidf)

# For Random Forest Classifier
rf_classifier = joblib.load('random_forest_classifier_model.pkl')
rf_pred = rf_classifier.predict(new_text_tfidf)

# For Gradient Boosting Machine
gbm_classifier = joblib.load('Gradient_Boosting_classifier_model.pkl')
gbm_pred = gbm_classifier.predict(new_text_tfidf)

# Print predictions
print("Logistic Regression Prediction:", logistic_pred)
print("SVM Prediction:", svm_pred)
print("Random Forest Prediction:", rf_pred)
print("Gradient Boosting Prediction:", gbm_pred)
