from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import re


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fare_medians_ = None
        self.age_medians_ = None
        self.surname_counts_ = None
        self.deck_model_ = None
        self.drop_columns_ = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        self.rare_titles = ['Don', 'Rev', 'Dr', 'Mme', 'Major',
                            'Capt', 'Col', 'Countess',
                            'Sir', 'Lady', 'Jonkheer']
        self.family_size_category_lambda = lambda size: (
            'Solo' if size == 1 else
            ('Small' if size <= 4 else
             ('Medium' if size <= 6 else
              'Large'
              )
             )
        )

    def fit(self, X, y=None):
        X = X.copy()

        # Fare by Pclass
        self.fare_medians_ = X.loc[X['Fare'] > 0].groupby('Pclass')[
            'Fare'].median()

        # Age by (Pclass, Sex)
        self.age_medians_ = {
            (1, 'male'): 36.0,
            (1, 'female'): 32.5,
            (2, 'male'): 29.0,
            (2, 'female'): 28.0,
            (3, 'male'): 28.0,
            (3, 'female'): 28.0
        }

        self.embarked_mode_ = X['Embarked'].mode()[0]

        # Surname group sizes
        surnames = X['Name'].apply(lambda name: name.split(',')[0].strip())
        self.surname_counts_ = surnames.value_counts()

        # Deck predictor
        X['Deck'] = X['Cabin'].apply(self.extract_deck)
        known_deck = X.loc[X['Deck'].notna()]
        if not known_deck.empty:
            self.deck_model_ = RandomForestClassifier(
                max_depth=4, random_state=42)
            self.deck_model_.fit(
                known_deck[['Pclass', 'Fare']],
                known_deck['Deck']
            )

        # Learn bin edges
        self.fare_bins_ = pd.qcut(
            X['Fare'], 5, retbins=True, duplicates='drop')[1]
        self.age_bins_ = pd.qcut(
            X['Age'], 5, retbins=True, duplicates='drop')[1]

        return self

    def transform(self, X):
        X = X.copy()

        # Fare Imputation
        def impute_fare(row):
            if pd.isnull(row['Fare']) or row['Fare'] == 0:
                return self.fare_medians_.get(row['Pclass'],
                                              X['Fare'].median())
            return row['Fare']
        X['Fare'] = X.apply(impute_fare, axis=1)

        X['Embarked'] = X['Embarked'].fillna(self.embarked_mode_)

        # Age Imputation
        def impute_age(row):
            if pd.isnull(row['Age']):
                key = (row['Pclass'], row['Sex'].lower())
                return self.age_medians_.get(key, X['Age'].median())
            return row['Age']
        X['Age'] = X.apply(impute_age, axis=1)

        # Family Size
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['FamilySizeCategory'] = X['FamilySize'].apply(
            self.family_size_category_lambda
        )

        # Ticket Prefix and Group Size
        # Simple prefix extraction with PC and CA as valid prefixes
        # There are 31 categories for TicketPrefix, reduced to 4:
        # PC, CA, Rare, Missing
        def extract_ticket_prefix(ticket):
            if pd.isnull(ticket):
                return 'Missing'
            ticket = ticket.replace('/', '').replace('.', '').strip()
            parts = ticket.split()

            if len(parts) > 1:
                prefix = parts[0]
            else:
                prefix = 'NoPrefix' if ticket.isdigit() else ticket.upper()
            return prefix if prefix in ['PC', 'CA'] else 'Rare'

        X['TicketPrefix'] = X['Ticket'].apply(extract_ticket_prefix)
        ticket_counts = X['Ticket'].value_counts()
        X['TicketGroupSize'] = X['Ticket'].map(ticket_counts).fillna(1)

        # Cabin Deck
        X['Deck'] = X['Cabin'].apply(self.extract_deck)
        missing_deck = X['Deck'].isna()
        if missing_deck.any() and self.deck_model_ is not None:
            X.loc[missing_deck, 'Deck'] = self.deck_model_.predict(
                X.loc[missing_deck, ['Pclass', 'Fare']]
            )
        X['Deck'] = X['Deck'].fillna('UnknownDeck')

        # Name Features
        X['Title'] = X['Name'].apply(self.extract_title)
        X['Surname'] = X['Name'].apply(lambda name: name.split(',')[0].strip())

        # Apply FareBin
        if self.fare_bins_ is not None:
            X['FareBin'] = pd.cut(X['Fare'], bins=self.fare_bins_,
                                  labels=False, include_lowest=True)

        # Apply AgeBin
        if self.age_bins_ is not None:
            X['AgeBin'] = pd.cut(X['Age'], bins=self.age_bins_,
                                 labels=False, include_lowest=True)

        X['TicketGroupSizeBin'] = X['TicketGroupSize'].apply(
            self.ticket_group_size_bin)

        X['Sex_Pclass'] = X['Sex'].astype(str) + '_' + X['Pclass'].astype(str)

        # Drop Unnecessary Columns
        X = X.drop(columns=self.drop_columns_, errors='ignore')

        return X

    def extract_deck(self, cabin):
        if pd.isnull(cabin):
            return np.nan
        cabin = str(cabin).strip().split(' ')[0]
        return cabin[0] if cabin else np.nan

    def extract_title(self, name):
        titles = "Mrs|Mr|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Capt|Col|Countess"
        pattern = r"({titles})".format(titles=titles)
        match = re.search(pattern, name)
        title = match.group() if match else 'Other'
        return 'RareTitle' if title in self.rare_titles else title

    def ticket_group_size_bin(self, size):
        if size == 1:
            return 'Solo'
        elif size <= 4:
            return 'Small'
        else:
            return 'Large'
