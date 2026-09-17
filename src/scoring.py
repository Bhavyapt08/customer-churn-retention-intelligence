import pandas as pd


LOW_THRESHOLD = 0.2363
MEDIUM_THRESHOLD = 0.5217
HIGH_THRESHOLD = 0.7084
HIGH_VALUE_THRESHOLD = 1960.30


def assign_risk_segment(probability):

    if probability <= LOW_THRESHOLD:
        return "Low"

    elif probability <= MEDIUM_THRESHOLD:
        return "Medium"

    elif probability <= HIGH_THRESHOLD:
        return "High"

    else:
        return "Critical"


def assign_retention_priority(
    risk_segment,
    monetary_value
):

    high_value_customer = (
        monetary_value >= HIGH_VALUE_THRESHOLD
    )

    if (
        risk_segment == "Critical"
        and high_value_customer
    ):
        return "Immediate Action"

    elif risk_segment == "Critical":
        return "High Priority"

    elif (
        risk_segment == "High"
        and high_value_customer
    ):
        return "High Priority"

    elif risk_segment == "High":
        return "Medium Priority"

    else:
        return "Monitor"


def recommend_action(retention_priority):

    if retention_priority == "Immediate Action":
        return (
            "Personalised retention outreach "
            "with high-value incentive"
        )

    elif retention_priority == "High Priority":
        return (
            "Targeted re-engagement campaign "
            "with personalised offer"
        )

    elif retention_priority == "Medium Priority":
        return (
            "Automated re-engagement reminder "
            "or promotional campaign"
        )

    else:
        return (
            "Continue monitoring and regular "
            "customer engagement"
        )


def score_customers(
    model,
    customer_data,
    model_features
):

    scored = customer_data.copy()

    probabilities = model.predict_proba(
        scored[model_features]
    )[:, 1]

    scored["ChurnProbability"] = probabilities

    scored["RiskSegment"] = (
        scored["ChurnProbability"]
        .apply(assign_risk_segment)
    )

    scored["HighValueCustomer"] = (
        scored["MonetaryValue"]
        >= HIGH_VALUE_THRESHOLD
    ).astype(int)

    scored["RetentionPriority"] = (
        scored.apply(
            lambda row:
            assign_retention_priority(
                row["RiskSegment"],
                row["MonetaryValue"]
            ),
            axis=1
        )
    )

    scored["RecommendedAction"] = (
        scored["RetentionPriority"]
        .apply(recommend_action)
    )

    return scored


def score_single_customer(
    model,
    customer_features,
    model_features
):

    input_df = pd.DataFrame(
        [customer_features]
    )

    probability = float(
        model.predict_proba(
            input_df[model_features]
        )[:, 1][0]
    )

    risk_segment = (
        assign_risk_segment(
            probability
        )
    )

    monetary_value = float(
        customer_features[
            "MonetaryValue"
        ]
    )

    retention_priority = (
        assign_retention_priority(
            risk_segment,
            monetary_value
        )
    )

    recommended_action = (
        recommend_action(
            retention_priority
        )
    )

    return {
        "churn_probability": probability,
        "risk_segment": risk_segment,
        "retention_priority": retention_priority,
        "recommended_action": recommended_action
    }