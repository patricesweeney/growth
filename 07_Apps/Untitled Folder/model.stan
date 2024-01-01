data {
    int<lower=0> N;  // Number of observations
    int<lower=1> J;  // Number of potential outcomes
    array[N] real price;  // Regressor (price)
    array[N] int<lower=1, upper=J> y;  // Observed outcomes
}

parameters {
    array[J] real alpha;  // Intercepts for each outcome
    vector<upper=0>[N] beta;  // Coefficients for price for each observation, constrained to be negative
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

    // Priors for varying intercepts and individual betas
    alpha ~ normal(mu_alpha, sigma_alpha);
    for (n in 1:N) {
        beta[n] ~ normal(mu_beta, sigma_beta);  // Prior for each individual beta[n]
    }

    // Likelihood
    for (n in 1:N) {
        vector[J] utilities;
        for (j in 1:J) {
            utilities[j] = alpha[j] + beta[n] * price[n];  // Use individual-specific beta[n]
        }
        y[n] ~ categorical_logit(utilities);
    }
}

generated quantities {
    matrix[N, J] choice_probabilities;  // Probabilities for each choice over all observations
    vector[J] alpha_out = alpha;  // Saving the alpha parameters
    vector[N] beta_out = beta;  // Saving the beta parameters
    real mu_alpha_out = mu_alpha;  // Saving the mu_alpha hyperparameter
    real sigma_alpha_out = sigma_alpha;  // Saving the sigma_alpha hyperparameter
    real mu_beta_out = mu_beta;  // Saving the mu_beta hyperparameter
    real sigma_beta_out = sigma_beta;  // Saving the sigma_beta hyperparameter

    for (n in 1:N) {
        vector[J] utilities;
        for (j in 1:J) {
            utilities[j] = alpha[j] + beta[n] * price[n];  // Use individual-specific beta[n]
        }
        choice_probabilities[n] = to_row_vector(softmax(utilities));
    }
}
