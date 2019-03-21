# Présenté par Ahmed Mohamed et Boumedienne Boukharouba

## Solve Every Sudoku Puzzle

## See http://norvig.com/sudoku.html

## Throughout this program we have:
##   r is a row,    e.g. 'A'
##   c is a column, e.g. '3'
##   s is a square, e.g. 'A3'
##   d is a digit,  e.g. '9'
##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
##   grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
##   values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}


import random, math


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s], [])) - set([s]))
             for s in squares)
counter = 0


################ Unit Tests ################

def test():
    "A set of tests that must pass."
    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                               'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                               'A1', 'A3', 'B1', 'B3'])
    print('All tests pass.')


################ Parse a Grid ################

def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    ## To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    print(grid_values(grid))
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  ## (Fail if we can't assign d to square s.)

    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


################ Constraint Propagation ################

def assign(values, s, d, count=0):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    global counter
    counter += 1
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  ## Already eliminated
    values[s] = values[s].replace(d, '')
    ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  ## Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    ## (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  ## Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values


################ Display as 2-D grid ################

def display(values):
    "Display these values as a 2-D grid."
    width = 1 + max(len(values[s]) for s in squares)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    print()


################ Search ################

def solve(grid): return search_hill_climbing(parse_grid(grid))


# Cette fonction serve à intializer la résolution de la grille avec la méthode de hill climbing
def search_hill_climbing(values):
    global counter
    conflicts, sum, counter = float('inf'), 0, 0
    final, values_copy = [], values.copy()
    tuples = []
    counter_nodes = 0

    # Tant que le nombre de conflits est supérieur à 0
    while conflicts > 0:

        # Trouve le changement avec le nombre minimal de conflits et le fixe dans la grille
        values_copy, sum, swap, counter_nodes = calculate_hill(values_copy, conflicts, tuples, counter_nodes)
        tuples.remove(swap)         # Enleve le chengement de la liste des changements possibles
        final = values_copy         # Valeur final qui sera tourné si jamais on trouve une solution pour la grille
        counter += counter_nodes

        # Si on atteint un maximum local (nombre de conflits du prochain changement est plus élevé que le courant)
        if sum > conflicts:
            display(values)
            print("Maximum Local\nConflits : ", conflicts, "\nNoeuds Explorés : ", counter_nodes,
                  "\n--------------------------------\n")
            return False

        # Sinon on continue
        conflicts = sum

    # Si l'algo trouve une solution pour la grille
    display(values)
    print("Noeuds Explorés : ", counter, "\n--------------------------------\n")
    return final


# Cette fonction calcule le nombre de conflits générés par un changement possible des deux éléments de la grille (swap)

def calculate_hill(values, conflicts, tuples, counter_nodes):
    conflicts_dict = {}
    fixed_values = [k for k, v in values.items() if len(v) == 1]
    square_peers = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

    # Intialize la grille avec des valeurs aléatoires
    values = initialize_randomly(values, square_peers)

    # Rempli la liste tuples avec tous les changements possibles des cellules
    if len(tuples) == 0:
        tuples.extend(list([(p[i], x) for p in square_peers for i in range(len(p)) for x in p[i + 1:]]))

    # Teste les conflits pour les changements possibles et prend celui avec le nombre minimale de conflits
    for t in tuples:
        new_values = values.copy()
        if (new_values[t[0]] and new_values[t[1]]) not in fixed_values:
            new_values[t[0]], new_values[t[1]] = new_values[t[1]], new_values[t[0]]
            sum_conflicts = calculate_conflicts(new_values)
            conflicts_dict[t] = sum_conflicts
            counter_nodes += 1
    swap, min_conflict = min(conflicts_dict.items(), key=lambda x: x[1])    # Choisi le changement avec le nombre minimal de conflits

    # retourne le nombre minimal de conflits si jamais il est inférieur au nombre de conflits courant
    if min_conflict < conflicts:
        values[swap[0]], values[swap[1]] = values[swap[1]], values[swap[0]]
    return values, min_conflict, swap, counter_nodes


################ Initializtion de la grille ################

def initialize_randomly(values, square_peers):
    peer = []
    values_dict = {(k, v) for k, v in values.items() if len(v) > 1}

    for k, v in values_dict:
        for z in square_peers:
            if k in z:
                peer = z
        for y in range(10):
            if str(y) not in [values[t] for t in peer if t != k]:
                values[k] = str(y)

    return values


################ Estimation de conflits ################
# Cette fonction serve à calculer le nombre de conflits par grille

def calculate_conflicts(values):
    conflicts_dict = {}
    for s in values:
        conflicts = 0
        for x in peers[s]:
            if values[x] == values[s]:
                conflicts += 1
                conflicts_dict[s] = conflicts

    return int(sum(conflicts_dict.values()) / 2)


def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False


def from_file(filename, sep='\n'):
    "Parse a file into a list of strings, separated by sep."
    return file(filename).read().strip().split(sep)


def shuffled(seq):
    "Return a randomly shuffled copy of the input sequence."
    seq = list(seq)
    random.shuffle(seq)
    return seq


################ System test ################

import time, random


def solve_all(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""

    def time_solve(grid):
        start = time.clock()
        values = solve(grid)
        t = time.clock() - start
        print(t, "\n")
        ## Display puzzles that take long enough
        if showif is not None and t > showif:
            display(grid_values(grid))
            if values: display(values)
            print('(%.2f seconds)\n' % t)
        return (t, solved(values))

    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs)" % (
            sum(results), N, name, sum(times) / N, N / sum(times), max(times)))
        print("\nCounter", counter, "\nMoyenne ", counter / N)


def solved(values):
    "A puzzle is solved if each unit is a permutation of the digits 1 to 9."

    def unitsolved(unit): return set(values[s] for s in unit) == set(digits)

    return values is not False and all(unitsolved(unit) for unit in unitlist)


def random_puzzle(N=17):
    """Make a random puzzle with N or more assignments. Restart on contradictions.
    Note the resulting puzzle is not guaranteed to be solvable, but empirically
    about 99.8% of them are solvable. Some have multiple solutions."""
    values = dict((s, digits) for s in squares)
    for s in shuffled(squares):
        if not assign(values, s, random.choice(values[s])):
            break
        ds = [values[s] for s in squares if len(values[s]) == 1]
        if len(ds) >= N and len(set(ds)) >= 8:
            return ''.join(values[s] if len(values[s]) == 1 else '.' for s in squares)
    return random_puzzle(N)  ## Give up and make a new puzzle


grid1 = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
grid2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
hard1 = '.....6....59.....82....8....45........3........6..3.54...325..6..................'
hardest = '8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..'

# display(parse_grid(grid2))

if __name__ == '__main__':
    test()
    solve_all([grid1], "easy", None)
    solve_all([hard1], "hard", None)
    solve_all([random_puzzle() for _ in range(99)], "random", 100.0)

## References used:
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/
