data {
    int<lower=0> N;  // Number of observations
    int<lower=1> J;  // Number of potential outcomes
    array[N] real price;  // Regressor (price)
}

parameters {
    array[J] real alpha;  // Intercepts for each outcome
    vector<upper=0>[N] beta;  // Coefficient for price for each observation, constrained to be negative
    real mu_alpha;  // Global mean for alpha distribution
    real<lower=0> sigma_alpha;  // Global standard deviation for alpha distribution
    real<upper=0> mu_beta;  // Mean for beta distribution, constrained to be negative
    real<lower=0> sigma_beta;  // Standard deviation for beta distribution
}

model {
    // Priors for hyperparameters
    mu_alpha ~ normal(0, 5);
    sigma_alpha ~ inv_gamma(2, 0.5);
    mu_beta ~ normal(0, 5);
    sigma_beta ~ inv_gamma(2, 0.5);

    // Priors for varying intercepts and betas
    alpha ~ normal(mu_alpha, sigma_alpha);
    for (n in 1:N)
        beta[n] ~ normal(mu_beta, sigma_beta);  // Prior for each beta[n], constrained to be negative

    // Likelihood removed for prior predictive sampling
    // for (n in 1:N) { ... }
}

generated quantities {
    matrix[N, J] choice_probabilities;  // Probabilities for each choice over all observations

    for (n in 1:N) {
        vector[J] utilities;
        for (j in 1:J) {
            utilities[j] = alpha[j] + beta[n] * price[n];  // Use individual-specific beta[n]
        }
        choice_probabilities[n] = to_row_vector(softmax(utilities));
    }
}
