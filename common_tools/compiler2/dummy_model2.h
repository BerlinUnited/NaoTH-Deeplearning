#ifndef DUMMY_MODEL2_H
#define DUMMY_MODEL2_H

#include <emmintrin.h>

#include "AbstractCNNClassifier.h"

class dummy_model2 : public AbstractCNNFinder {

public:
	void cnn(float x0[16][16][1]);
	void predict(const BallCandidates::PatchYUVClassified& p,double meanBrightness);
	virtual double getRadius();
	virtual Vector2d getCenter();
	virtual double getBallConfidence();

private:
	float in_step[16][16][1];
	float scores[4];

};
# endif
