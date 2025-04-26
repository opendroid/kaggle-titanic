from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import re
# 1. FareByPclassImputer


class FareByPclassImputer(BaseEstimator, TransformerMixin):
    """
    Replace Fare=0 with the median Fare of the corresponding Pclass.
    """

    def __init__(self, fare_column='Fare', pclass_column='Pclass'):
        self.fare_column = fare_column
        self.pclass_column = pclass_column
        self.pclass_median_fares_ = None

    def fit(self, X, y=None):
        self.pclass_median_fares_ = (
            X.loc[X[self.fare_column] > 0]
            .groupby(self.pclass_column)[self.fare_column]
            .median()
        )
        return self

    def transform(self, X):
        X = X.copy()

        def impute_fare(row):
            if row[self.fare_column] == 0:
                return self.pclass_median_fares_.get(row[self.pclass_column],
                                                     np.nan)
            return row[self.fare_column]

        X[self.fare_column] = X.apply(impute_fare, axis=1)
        return X

# 2. AgeByClassSexImputer


class AgeByClassSexImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing Age values based on Pclass and Sex-specific medians.
    """

    def __init__(self, age_column='Age', pclass_column='Pclass',
                 sex_column='Sex'):
        self.age_column = age_column
        self.pclass_column = pclass_column
        self.sex_column = sex_column
        self.medians_ = {
            (1, 'male'): 37.0,
            (1, 'female'): 35.0,
            (2, 'male'): 30.0,
            (2, 'female'): 28.0,
            (3, 'male'): 25.0,
            (3, 'female'): 21.5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def impute_age(row):
            if pd.isnull(row[self.age_column]):
                key = (row[self.pclass_column], row[self.sex_column].lower())
                return self.medians_.get(key, np.nan)
            return row[self.age_column]

        X[self.age_column] = X.apply(impute_age, axis=1)
        return X

# 3. FamilySizeAdder


class FamilySizeAdder(BaseEstimator, TransformerMixin):
    """
    Add FamilySize and optionally FamilySizeCategory based on SibSp and Parch.
    """

    def __init__(self, add_family_size_category=False):
        self.add_family_size_category = add_family_size_category

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        if self.add_family_size_category:
            X['FamilySizeCategory'] = X['FamilySize'].apply(
                self.family_size_category)

        return X

    def family_size_category(self, size):
        if size == 1:
            return 'Single'
        elif size <= 4:
            return 'Small'
        else:
            return 'Large'

# 4. TicketFeatureAdder


class TicketFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Extract TicketPrefix and TicketGroupSize from Ticket column.
    """

    def __init__(self, ticket_column='Ticket'):
        self.ticket_column = ticket_column
        self.ticket_counts_ = None

    def fit(self, X, y=None):
        self.ticket_counts_ = X[self.ticket_column].value_counts()
        return self

    def transform(self, X):
        X = X.copy()

        def extract_ticket_prefix(ticket):
            if pd.isnull(ticket):
                return 'Missing'
            ticket = ticket.replace('/', '').replace('.', '').strip()
            parts = ticket.split()
            if len(parts) > 1:
                return parts[0]
            else:
                if ticket.isdigit():
                    return 'NoPrefix'
                else:
                    return ticket.upper()

        X['TicketPrefix'] = X[self.ticket_column].apply(extract_ticket_prefix)
        X['TicketGroupSize'] = X[self.ticket_column].map(
            self.ticket_counts_).fillna(1)
        return X

# 5. CabinDeckExtractor


class CabinDeckExtractor(BaseEstimator, TransformerMixin):
    """
    Extract Deck from Cabin and predict missing Deck using
        DecisionTreeClassifier.
    """

    def __init__(self,
                 cabin_column='Cabin',
                 pclass_column='Pclass',
                 fare_column='Fare'):
        self.cabin_column = cabin_column
        self.pclass_column = pclass_column
        self.fare_column = fare_column
        self.model_ = None

    def fit(self, X, y=None):
        X = X.copy()
        X['Deck'] = X[self.cabin_column].apply(self.extract_deck)

        known_deck = X.loc[X['Deck'].notna()]
        if not known_deck.empty:
            self.model_ = DecisionTreeClassifier(max_depth=4, random_state=42)
            self.model_.fit(
                known_deck[[self.pclass_column, self.fare_column]],
                known_deck['Deck']
            )
        return self

    def transform(self, X):
        X = X.copy()
        X['Deck'] = X[self.cabin_column].apply(self.extract_deck)

        missing_deck = X['Deck'].isna()
        if missing_deck.any() and self.model_ is not None:
            X.loc[missing_deck, 'Deck'] = self.model_.predict(
                X.loc[missing_deck, [self.pclass_column, self.fare_column]]
            )

        return X

    def extract_deck(self, cabin_value):
        if pd.isnull(cabin_value):
            return np.nan
        cabin_value = str(cabin_value).strip().split(' ')[0]
        return cabin_value[0] if cabin_value else np.nan


class NameFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Extracts Title and Surname from Name column,
    and calculates SurnameGroupSize.
    """

    def __init__(self, name_column='Name'):
        self.name_column = name_column
        self.surname_counts_ = None

    def fit(self, X, y=None):
        surnames = X[self.name_column].apply(self.extract_surname)
        self.surname_counts_ = surnames.value_counts()
        return self

    def transform(self, X):
        X = X.copy()
        X['Title'] = X[self.name_column].apply(self.extract_title)
        X['Surname'] = X[self.name_column].apply(self.extract_surname)
        X['SurnameGroupSize'] = X['Surname'].map(
            self.surname_counts_).fillna(1)
        return X

    def extract_title(self, name):
        titles = "Mrs|Mr|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Capt|Col|Countess"
        pattern = r"({titles})".format(titles=titles)
        match = re.search(pattern, name)
        return match.group() if match else 'Other'

    def extract_surname(self, name):
        return name.split(',')[0].strip()
