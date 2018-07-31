from matrix_factor_model import ProductRecommender

    
# fake data. (2 users, three products)
user_1 = [1, 2, 3]
user_2 = [0, 2, 3]
user_3 = [1, 2, 0]
data = [user_1, user_2, user_3]
 
# train model
modelA = ProductRecommender()
modelA.fit(data,latent_features_guess=5, learning_rate=0.0002, steps=5000, regularization_penalty=0.02, convergeance_threshold=0.01)
 
# predict for user 2 
#print(modelA.predict_instance([1,2]))
print(modelA.predict_all())

