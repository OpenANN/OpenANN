#ifndef OPENANN_EVALUATION_H_
#define OPENANN_EVALUATION_H_

/**
 * @file Evaluation.h
 *
 * Provides some convenience functions to evaluate learners.
 */

#include <Eigen/Dense>

namespace OpenANN
{

class Optimizer;
class Learner;
class DataSet;

/**
 * Sum of squared errors.
 *
 * This error function is usually used for regression problems.
 *
 * \f$ E = \sum_n \sum_f (y^{(n)}_f - t^{(n)}_f)^2,
 * \f$ where \f$ y^{(n)}_f \f$ is the predicted output and \f$ t^{(n)}_f \f$
 * the desired output.
 *
 * @param learner learned model
 * @param dataSet dataset
 * @return SSE
 */
double sse(Learner& learner, DataSet& dataSet);

/**
 * Mean squared error.
 *
 * This error function is usually used for regression problems.
 *
 * \f$ E = \frac{1}{N} \sum_n \sum_f (y^{(n)}_f - t^{(n)}_f)^2,
 * \f$ where \f$ y^{(n)}_f \f$ is the predicted output, \f$ t^{(n)}_f \f$
 * the desired output and \f$ N \f$ is size of the dataset.
 *
 * @param learner learned model
 * @param dataSet dataset
 * @return MSE
 */
double mse(Learner& learner, DataSet& dataSet);

/**
 * Root mean squared error.
 *
 * This error function is usually used for regression problems.
 *
 * \f$ E = \sqrt(\frac{1}{N} \sum_n \sum_f (y^{(n)}_f - t^{(n)}_f)^2),
 * \f$ where \f$ y^{(n)}_f \f$ is the predicted output, \f$ t^{(n)}_f \f$
 * the desired output and \f$ N \f$ is size of the dataset.
 *
 * @param learner learned model
 * @param dataSet dataset
 * @return RMSE
 */
double rmse(Learner& learner, DataSet& dataSet);

/**
 * Cross entropy.
 *
 * This error function is usually used for classification problems.
 *
 * \f$ E = - \sum_n \sum_f t^{(n)}_f \log y^{(n)}_f, \f$
 * where \f$ t^{(n)}_f \f$ represents the actual probability
 * \f$ P(f|\bf x^{(n)}) \f$ and \f$ y^{(n)}_f \f$ is the prediction of the
 * learner.
 *
 * @param learner learned model
 * @param dataSet dataset
 */
double ce(Learner& learner, DataSet& dataSet);

/**
 * Accuracy.
 *
 * The percentage of correct predictions in a classification problem.
 *
 * @param learner learned model
 * @param dataSet dataset
 * @return accuracy, within [0, 1]
 */
double accuracy(Learner& learner, DataSet& dataSet);

/**
 * Confusion matrix.
 *
 * Requires one-of-c encoded labels. The matrix row will denote the actual
 * label and the matrix column will denote the predicted label of the learner.
 *
 * @param learner learned model
 * @param dataSet dataset
 * @return confusion matrix
 */
Eigen::MatrixXi confusionMatrix(Learner& learner, DataSet& dataSet);

/**
 * Classification hits.
 *
 * @param learner learned model
 * @param dataSet dataset; the targets are assumed to be encoded with 1-of-c
 *                encoding if there are 2 or more components, otherwise the
 *                output is assumed to be within [0, 1], where values of 0.5
 *                or greater and all other values belong to different classes
 * @return number of correct predictions
 */
int classificationHits(Learner& learner, DataSet& dataSet);

/**
 * Cross-validation.
 *
 * @param folds number of cross-validation folds
 * @param learner learner
 * @param dataSet dataset
 * @param opt optimization algorithm
 * @return average accuracy on validation set, within [0, 1]
 */
double crossValidation(int folds, Learner& learner, DataSet& dataSet,
                       Optimizer& opt);

/**
 * One-of-c decoding.
 *
 * @param target vector that represents a 1-of-c encoded class label
 * @return index of entry with the highest value
 */
int oneOfCDecoding(const Eigen::VectorXd& target);

} // namespace OpenANN

#endif // OPENANN_EVALUATION_H_
