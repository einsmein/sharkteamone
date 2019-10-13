import os
import shutil
from config import ENDPOINT, PREDICTION_KEY, PROJECT_ID

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

publish_iteration_name = "Iteration1"

base_image_path = "D:\\Dropbox\\Projects\\Shark Team One - photos - whale sharks - Jinnapat\\Organized"
test_image = os.path.join(base_image_path, "Unusable\\Test\\G0030945.jpg")

test_image_path = [
	os.path.join(base_image_path, "Unusable\\Test"),
	os.path.join(base_image_path, "Usable\\Fin intact\\Test"),
	os.path.join(base_image_path, "Usable\\Fin damage\\Test"),	
]


def predict(predictor, image_path):
	with open(image_path, "rb") as image_contents:
	    results = predictor.classify_image(
	        PROJECT_ID, publish_iteration_name, image_contents.read())

	    # Choose result with hight probability
	    prediction = [(prediction.probability, prediction.tag_name) for prediction in results.predictions]
	    prediction.sort()

	    # for prediction in results.predictions:
	    #     print("\t" + prediction.tag_name +
	    #           ": {0:.2f}%".format(prediction.probability * 100))

	    return prediction


def test_predict(predictor):
	result = predict(predictor, test_image)
	print(result)


def move_to_classified(predictor, image_path, threshold, classified_path):
	result = predict(predictor, image_path)
	if result[-1][0] > threshold:
		result_dir = os.path.join(classified_path, result)
		if not os.exists(result_dir):
			os.mkdir(result_dir)
		shutil.copy2(image_path, result_dir)


def main():
	predictor = CustomVisionPredictionClient(PREDICTION_KEY, endpoint=ENDPOINT)
	unclassified_image_path = ""
	classified_image_path = ""
	threshold = 0.8

	test_predict(predictor)

	# images = os.listdir(unclassified_image_path)
	# for image in images:
	# 	image_path = os.path.join(unclassified_image_path, image)

	# 	result = predict(image_path)
	# 	result_dir = os.path.join(classified_image_path, result)
	# 	if not os.exists(result_dir):
	# 		os.mkdir(result_dir)
	# 	shutil.copy2(image_path, result_dir)


if __name__ == "__main__":
	main()