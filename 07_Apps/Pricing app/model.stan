data {
    int<lower=0> N;  // Number of observations
    int<lower=1> J;  // Number of potential outcomes
    array[N] real price;  // Regressor (price)
    array[N] int<lower=1, upper=J> y;  // Observed outcomes
}

parameters {
    vector[J] alpha;  // Intercept for each potential outcome
    real<upper=0> beta;  // Single coefficient for price, constrained to be negative
    real mu_alpha;  // Global mean for alpha distribution
    real<lower=0> sigma_alpha;  // Global standard deviation for alpha distribution
    real mu_beta;  // Global mean for beta distribution
    real<lower=0> sigma_beta;  // Global standard deviation for beta distribution
}

model {
    // Priors for hyperparameters
    mu_alpha ~ normal(0, 5);
    sigma_alpha ~ inv_gamma(2, 0.5);
    mu_beta ~ normal(0, 5);
    sigma_beta ~ inv_gamma(2, 0.5);

    // Priors for varying intercepts
    alpha ~ normal(mu_alpha, sigma_alpha);

    // Single beta prior
    beta ~ normal(mu_beta, sigma_beta);  // Adjust the prior as needed

    // Likelihood
    for (n in 1:N) {
        vector[J] utilities;
        for (j in 1:J) {
            utilities[j] = alpha[j] + beta * price[n];  // Use the single beta
        }
        y[n] ~ categorical_logit(utilities);
    }
}

generated quantities {
    matrix[N, J] choice_probabilities;  // Probabilities for each choice over all observations

    for (n in 1:N) {
        vector[J] utilities;
        for (j in 1:J) {
            utilities[j] = alpha[j] + beta * price[n];
        }
        choice_probabilities[n] = to_row_vector(softmax(utilities));
    }
    // Direct output of alpha and beta parameters
    // No need for additional variables like saved_alpha or saved_beta
}
