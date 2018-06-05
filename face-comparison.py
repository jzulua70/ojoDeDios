import boto3

BUCKET = "amazon-rekognition"
KEY_SOURCE = "../../video/Diego.jpg"
KEY_TARGET = "../../video/Diego2.jpg"

def compare_faces(bucket, key, bucket_target, key_target, threshold=80, region="us-east-1"):
	rekognition = boto3.client("rekognition", region)
	source = "../../video/Diego.jpg"
	target = "../../video/Diego2.jpg"
	img1 = open(source, 'rb')
	img2 = open(target, 'rb')
	response = rekognition.compare_faces( SourceImage={'Bytes': img1.read()},TargetImage={'Bytes': img2.read()},SimilarityThreshold=threshold)

	return response['SourceImageFace'], response['FaceMatches']


source_face, matches = compare_faces(BUCKET, KEY_SOURCE, BUCKET, KEY_TARGET)

# the main source face
print "Source Face ({Confidence}%)".format(**source_face)

# one match for each target face
for match in matches:
	print "Target Face ({Confidence}%)".format(**match['Face'])
	print "  Similarity : {}%".format(match['Similarity'])

"""
    Expected output:
    
    Source Face (99.945602417%)
    Target Face (99.9963378906%)
      Similarity : 89.0%
"""