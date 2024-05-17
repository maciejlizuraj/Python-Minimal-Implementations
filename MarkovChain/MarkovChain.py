"""Rock-Paper-Scissors game against a Markov Chain AI opponent

This module provides a command-line Rock-Paper-Scissors game where the user plays against an AI
opponent. The opponent's choices are based on a transition count matrix that is updated throughout
the game. The game keeps track of the score and determines the winner at the end.

The module contains the following classes:

    Choice: An enumeration representing the possible choices (rock, paper, scissors) in the game.
    Outcome: An enumeration representing the outcome (win, tie, loss) of a round in the game.
    RoundEvaluator: A class that evaluates the outcome of a round based on the choices of the user
    and the opponent.
    Result: A class that represents the result of a round, including the choice made and the
    outcome. It also provides methods to convert the result to an integer and vice versa.
    Config: A class that stores configuration variables for the game, such as the maximum number of
    rounds and the maximum number of points to win.
    ValuesCounter: A class that keeps track of the current game state, including the current score,
    current round number, and methods to check if the game should continue and update the score
    based on the outcome of a round.
    Opponent: A class that represents the AI opponent in the game. It keeps track of a transition
    count matrix that stores how often the opponent chose a particular choice after a specific
    previous choice and outcome. The opponent uses this matrix to make its choices in the game.
    Game: A class that manages the overall flow of the game. It creates instances of other classes,
    handles user input, calls methods to evaluate rounds and update the game state, and prints
    messages to the console. It also allows loading and saving the opponent's transition count
    matrix to a file.

"""

import os.path
from enum import Enum
from typing import Dict, Tuple
import numpy as np


class Choice(Enum):
    """Represents the possible choices in Rock-Paper-Scissors game

    This enumeration defines the three possible choices a player can make in a game of
    Rock-Paper-Scissors:
    * ROCK (value 0)
    * PAPER (value 1)
    * SCISSORS (value 2)
    """
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class Outcome(Enum):
    """Represents the outcome of a round in Rock-Paper-Scissors game

    This enumeration defines the three possible outcomes of a round in a game of
    Rock-Paper-Scissors:
    * WIN (value 0)
    * TIE (value 1)
    * LOSS (value 2)
    """
    WIN = 0
    TIE = 1
    LOSS = 2


class RoundEvaluator:
    """Evaluates the outcome of a round in Rock-Paper-Scissors game

    This class provides a static method to evaluate the outcome of a round in a game of
    Rock-Paper-Scissors based on the choices of the user and the opponent.

    Attributes:
        choice_to_choice_beating_it_dict (dict): A dictionary that maps a Choice to the Choice that
        beats it.
    """
    choice_to_choice_beating_it_dict = {Choice.ROCK: Choice.PAPER, Choice.PAPER: Choice.SCISSORS,
                                        Choice.SCISSORS: Choice.ROCK}

    @classmethod
    def evaluate_round(cls, user_choice: Choice, opponent_choice: Choice) -> Outcome:
        """Evaluates the outcome of a round in Rock-Paper-Scissors game

        This method takes the choices of the user and the opponent and returns the outcome of the
        round.

        Attributes:
            user_choice (Choice): The choice made by the user.
            opponent_choice (Choice): The choice made by the opponent.

        Returns:
            Outcome: The outcome of the round (WIN, TIE, or LOSS).
        """
        outcome = Outcome.LOSS
        if user_choice == opponent_choice:
            outcome = Outcome.TIE
        elif RoundEvaluator.choice_to_choice_beating_it_dict[opponent_choice] == user_choice:
            outcome = Outcome.WIN
        return outcome


class Result:
    """Represents the result of a round in Rock-Paper-Scissors game

    This class is a wrapper for choice and outcome of a round. It also provides methods to translate
     the result into an integer which corresponds to index in a vector and vice versa.

    Attributes:
        choice (Choice): The choice made in the round.
        outcome (Outcome): The outcome of the round (WIN, TIE, or LOSS).
    """

    def __init__(self, choice: Choice, outcome: Outcome):
        """Initializes a Result object

        This constructor can be called with the choice and outcome.

        Args:
            choice (Choice): The choice made by the player in the round.
            outcome (Outcome): The outcome of the round.
        """
        self.choice = choice
        self.outcome = outcome

    def translate_to_int(self) -> int:
        """Translates the result into an integer

        This method combines the value of the choice (0, 1, or 2) and the outcome (0, 1, or 2) into
        a single integer.

        Returns:
        int: An integer representation of the result (choice * 3 + outcome).
        """
        return self.choice.value * 3 + self.outcome.value

    @staticmethod
    def translate_from_int(int_value: int) -> Tuple[Choice, Outcome]:
        """Converts an integer representation of the result back to choice and outcome.

        This method takes an integer value (obtained from translate_to_int) and separates it back
        into the original choice and outcome. The integer is divided by 3 (floor division) to get
        the choice index and the remainder is used to get the outcome index.

        Returns:
        Tuple[Choice, Outcome]: A tuple containing the choice and outcome extracted from the
        integer.
        """
        return Choice(int_value // 3), Outcome(int_value % 3)


class Config:
    """Holds configuration variables for the game

    This class stores configuration variables used throughout the game, such as the maximum number
    of rounds, maximum number of points to win, and the file path for storing the opponent's
     transition matrix.
    """
    max_rounds: int = 30
    max_points: int = 10
    matrix_file_path: str = "matrix.txt"


class ValuesCounter:
    """Keeps track of the current game state

    This class keeps track of the current number of rounds played, the current score, and provides
    methods to check if the game should continue and update the score based on the outcome of a
    round.

    Attributes:
        current_points (int): The current score of the player.
        current_round (int): The current round number.
        outcome_dict (dict): A dictionary that maps an Outcome to its corresponding point value
        (WIN: 1, TIE: 0, LOSS: -1).
    """

    def __init__(self, current_points: int = 0, current_round: int = 0):
        """Initializes a ValuesCounter object

        This constructor initializes the current score, current round number, and the dictionary
        that maps outcomes to
        point values.
        """
        outcome_dict = {Outcome.WIN: 1, Outcome.TIE: 0, Outcome.LOSS: -1}
        self.current_points = current_points
        self.current_round = current_round
        self.outcome_dict = outcome_dict

    def should_continue(self) -> bool:
        """Checks if the game should continue

        This method checks if the absolute value of the current score is less than the maximum
        number of points and if the current round number is less than the maximum number of rounds.
        The game continues as long as both conditions are true.

        Returns:
            bool: True if the game should continue, False otherwise.
        """
        return abs(
            self.current_points) < Config.max_points and self.current_round < Config.max_rounds

    def update_points(self, outcome: Outcome) -> None:
        """Updates the current score based on the outcome of a round

        This method adds the point value associated with the outcome
        (WIN - 1, TIE - 0, or LOSS - (-1)) to the current score.

        Args:
            outcome (Outcome): The outcome of the round.
        """
        self.current_points += self.outcome_dict[outcome]

    def print_current_game_state(self) -> None:
        """Prints the current game state

        This method prints the current round number and the current score to the console.
        """
        print(f'Current round {self.current_round}')
        print(f'Current points {self.current_points}')

    def print_game_over_info(self) -> None:
        """Prints the game over message based on the game state

        This method checks the current score and prints a message informing the user
        whether they won, lost, or tied the game. It also displays the final round count
        and score.
        """

        if self.current_points == 0:
            string = 'it is a tie'
        elif self.current_points < 0:
            string = 'you lost'
        else:
            string = 'you won'
        print(f'Game over! Game ended at round {self.current_round - 1}.\n'
              f'Points balance is {self.current_points} which means that {string}.')


class Opponent:
    """Represents the AI opponent in the game

    This class represents the AI opponent in the game. It keeps track of a transition count matrix
    that stores how often the opponent chose a particular choice after a specific previous choice
    and outcome. The opponent uses this matrix to make its choices in the game.

    Attributes:
        transition_count_matrix (np.ndarray): A numpy array that stores the transition count matrix.
        last_result (Result): The result of the last round.
    """

    def __init__(self, last_result: Result = Result(Choice.ROCK, Outcome.WIN)):
        """Initializes an Opponent object

        This constructor initializes the transition count matrix with zeros and the last result
        with the provided values (default: ROCK and WIN).

        Args:
            last_result (Result, optional): The result of the last round. Default value is for first
            game when there is no reference.
        """

        self.transition_count_matrix: np.ndarray = np.full(
            (len(Choice) * len(Outcome), len(Choice)), 0)
        self.last_result: Result = last_result

    def make_a_choice(self) -> Choice:
        """Makes a choice based on the transition count matrix

        This method selects a choice for the opponent based on the transition count matrix and the
        result of the last round. It considers the most frequent choice made by the opponent in the
        past after the same outcome for the previous user choice.

        Returns:
            Choice: The choice made by the opponent.
        """
        matrix_of_last_result = self.transition_count_matrix[self.last_result.translate_to_int()]
        max_count = matrix_of_last_result.max()
        possible_choices = np.where(matrix_of_last_result == max_count)[0]
        if len(possible_choices) == 0:
            choice = possible_choices[0]
        else:
            choice = np.random.choice(possible_choices)
        return RoundEvaluator.choice_to_choice_beating_it_dict[Choice(choice)]

    def update_the_matrix(self, new_result: Result) -> None:
        """Updates the transition count matrix

        This method updates the transition count matrix based on the user's choice, the opponent's
        choice, and the outcome of the round. It increments the count for the opponent's choice made
        after the user's choice and the previous round's outcome.

        Args:
            new_result (Result): The result of the current round.
        """
        self.transition_count_matrix[self.last_result.translate_to_int()][
            new_result.choice.value] += 1
        self.last_result = new_result

    def set_matrix(self, matrix: np.ndarray) -> None:
        """Sets the transition count matrix from a file

        This method allows setting the transition count matrix. It checks if the matrix has the
        correct size.

        Args:
            matrix (np.ndarray): The transition count matrix to be set.

        Raises:
            ValueError: If the given matrix does not have the required size.
        """
        if matrix.shape != (len(Choice) * len(Outcome), len(Choice)):
            raise ValueError("Matrix from the file does not have the required size")
        self.transition_count_matrix = matrix


class Game:
    """Manages the overall game flow

    This class manages the overall flow of the game. It creates instances of other classes like
    ValuesCounter and Opponent, handles user input, calls methods to evaluate rounds and update the
    game state, and prints messages to the console. It also allows loading and saving the opponent's
    transition count matrix to a file.
    """

    def __init__(self):
        """Initializes a Game object

        This constructor creates instances of ValuesCounter and Opponent, checks if the file
        containing the transition count matrix exists, and prints a message informing the user that
        the first round won't affect the scoring or the matrix update.
        """

        values_counter = ValuesCounter()
        opponent = Opponent()
        if self.file_exists():
            opponent.set_matrix(self.read_file())

        while values_counter.should_continue():
            current_choice = Game.get_input()
            print(f'You chose {current_choice.name}')

            opponent_choice = opponent.make_a_choice()
            print(f'Opponent chose {opponent_choice.name}')

            outcome = RoundEvaluator.evaluate_round(current_choice, opponent_choice)
            if values_counter.current_round != 0:
                opponent.update_the_matrix(Result(current_choice, outcome))
                values_counter.update_points(outcome)
            else:
                opponent.last_result = Result(current_choice, outcome)
                print(
                    'Due to lack of previous state round 0 does not count towards scoring or '
                    'change the matrix')

            values_counter.print_current_game_state()
            values_counter.current_round += 1
        values_counter.print_game_over_info()
        self.save_file(opponent.transition_count_matrix)

    @staticmethod
    def save_file(arr: np.ndarray) -> None:
        """Saves the transition count matrix to a file

        This static method saves the transition count matrix to the matrix file. It first
        adjusts the matrix by subtracting the minimum value in each row (to avoid saving unnecessary
        offset values) and then saves it to the file using numpy's `savetxt` function with delimiter
        ";".

        Args:
            arr (np.ndarray): The transition count matrix to be saved.
        """
        for m in range(arr.shape[0]):
            if np.all(arr[m, :] > 0):
                min_value = np.min(arr[m, :])
                arr[m, :] -= min_value
        arr = np.clip(arr, 0, 1000)
        np.savetxt(Config.matrix_file_path, arr, delimiter=';')

    @staticmethod
    def read_file() -> np.ndarray:
        """Reads the transition count matrix from a file

        This static method reads the transition count matrix from the matrix file.
        It uses numpy's `loadtxt` function with delimiter ";".

        Returns:
            np.ndarray: The transition count matrix loaded from the file.
        """

        data = np.loadtxt(Config.matrix_file_path, delimiter=';')
        return data

    @staticmethod
    def file_exists() -> bool:
        """Checks if the file containing the transition count matrix exists

        This static method checks if the matrix file exists.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.exists(Config.matrix_file_path)

    @staticmethod
    def get_input() -> Choice:
        """Gets user input for their choice

        This function prompts the user to enter their choice (rock, paper, or scissors) and
        validates the input before returning the corresponding Choice enum value.

        Returns:
            Choice: The user's choice.
        """
        choice_dictionary: Dict[str, Choice] = {"R": Choice.ROCK,
                                                "P": Choice.PAPER,
                                                "S": Choice.SCISSORS}
        choice = (input(
            "Input your choice. 'R' or 'r' for rock, 'P' or 'p' for paper, 'S' or 's' for scissors")
                  .strip().upper())
        if choice not in choice_dictionary:
            choice = 'R'
            print("Invalid choice. Defaulting to rock")
        return choice_dictionary[choice]


if __name__ == '__main__':
    Game()
