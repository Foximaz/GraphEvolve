import sys
sys.path.append("../../AGoTI")

from typing import Optional, Iterable, List, Dict
import logging
import sqlite3
import json
from collections import OrderedDict

from AGoTI.goo import GraphOfOperations
from AGoTI.utils import OrientedGraphNode

from .utils import SortedList
from .goo import GooConfig, parse_goo_json

logger = logging.getLogger(__name__)

class Solution(OrientedGraphNode):
    id_counter: int = 0
    
    def __init__(
            self,
            goo_json: Dict,
            score: float,
            feedback: str, # unused for now
            plan_draft: str,
            plan_formal: str,
            goos: List[GraphOfOperations],
            scores: List[float],
            parents: Optional[Iterable]=None,
            id: Optional[int]=None,
            loaded: bool=True
        ):
        super().__init__(parents, None, id=id) # evolution graph is acyclic
        self.goo_json = goo_json
        self.score = score
        self.feedback = feedback
        self.plan_draft = plan_draft
        self.plan_formal = plan_formal
        self.goos = goos
        self.scores = scores
        self.loaded = loaded

    def save(self, cursor: sqlite3.Cursor):
        cursor.execute(
            "INSERT INTO solutions (solution_id, goo_template, agg_score, agg_feedback, plan_draft, plan_formal) VALUES (?, ?, ?, ?, ?, ?)",
            (self.id, json.dumps(self.goo_json), self.score, self.feedback, self.plan_draft, self.plan_formal)
            )
        for parent in self.parents:
            cursor.execute(
                "INSERT INTO solutions_tree (solution_id, parent_id) VALUES (?, ?)",
                (self.id, parent.id)
                )
        for sample_id, (goo, score) in enumerate(zip(self.goos, self.scores)):
            cursor.execute(
                "INSERT INTO runs (solution_id, sample_id, got, score) VALUES (?, ?, ?, ?)",
                (self.id, sample_id, json.dumps(goo.get_got_json()), score)
                )

    def unload(self):
        self.loaded = False
        self.goo_json = None
        self.feedback = None
        self.plan_draft = None
        self.plan_formal = None
        self.goos = None
        self.scores = None

    def load(self, goo_config: GooConfig, cursor: sqlite3.Cursor):
        cursor.execute(
            "SELECT goo_template, agg_score, agg_feedback, plan_draft, plan_formal FROM solutions WHERE solution_id = ?",
            (self.id,)
        )
        result = cursor.fetchone()
        
        if result is None:
            logger.error(f"solution with id {self.id} not found in database")
            raise ValueError(f"Solution with id {self.id} not found in database")
        
        self.goo_json = json.loads(result[0])
        self.score = result[1]
        self.feedback = result[2]
        self.plan_draft = result[3]
        self.plan_formal = result[4]

        cursor.execute(
            """SELECT r.sample_id, r.got, r.score 
            FROM runs r
            WHERE r.solution_id = ?
            ORDER BY r.sample_id""",
            (self.id,)
        )
        all_runs = cursor.fetchall()
        
        goos = []
        scores = []
        
        for _, got_json_str, score in all_runs:
            got_json = json.loads(got_json_str)
            goo_object = parse_goo_json(self.goo_json, goo_config, got_json)
            goos.append(goo_object)
            scores.append(score)
        self.goos = goos
        self.scores = scores
        
        self.loaded = True

    def copy(self):
        pass


class SolutionDatabase:
    def __init__(
        self,
        goo_config,
        max_loaded_solutions: int = 10,
        db_path: str = "./solution_database.db",
        clear_db: bool = False,
    ):
        self.goo_config = goo_config
        self.solutions = SortedList()
        self.id_to_solution = {}
        self.best_solution = None

        self.max_loaded_solutions = max_loaded_solutions
        self.loaded_cache = OrderedDict()

        self._init_sqlite_db(db_path)

        if clear_db:
            self._clear_all_tables()
        else:
            self._load_existing_db()

    def _init_sqlite_db(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                solution_id INTEGER PRIMARY KEY,
                goo_template TEXT NOT NULL,
                agg_score REAL,
                agg_feedback TEXT,
                plan_draft TEXT,
                plan_formal TEXT
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions_tree (
                solution_id INTEGER,
                parent_id INTEGER,
                PRIMARY KEY (solution_id, parent_id)
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                solution_id INTEGER,
                sample_id INTEGER,
                got TEXT,
                score REAL,
                PRIMARY KEY (solution_id, sample_id)
            )
        """)

    def _clear_all_tables(self):
        self.conn.commit()
        for table in ["solutions", "solutions_tree", "runs"]:
            self.cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def _load_existing_db(self):
        self.cursor.execute("""
            SELECT solution_id, agg_score
            FROM solutions
        """)

        id_to_solution = {}

        for solution_id, score in self.cursor.fetchall():
            solution = Solution(
                goo_json=None,
                score=score,
                feedback=None,
                plan_draft=None,
                plan_formal=None,
                goos=None,
                scores=None,
                parents=None,
                id=solution_id,
                loaded=False,
            )
            id_to_solution[solution_id] = solution

        self.cursor.execute("SELECT solution_id, parent_id FROM solutions_tree")
        for solution_id, parent_id in self.cursor.fetchall():
            id_to_solution[solution_id].add_parents(
                [id_to_solution[parent_id]]
            )

        self.cursor.execute("""
            SELECT s.solution_id
            FROM solutions s
            LEFT JOIN solutions_tree st ON s.solution_id = st.solution_id
            WHERE st.solution_id IS NULL
        """)
        roots = [row[0] for row in self.cursor.fetchall()]
        if roots:
            self.root_solution = id_to_solution[roots[0]]

        for solution in id_to_solution.values():
            self.solutions.add(solution, (solution.score, solution.id))

        self.cursor.execute("""
            SELECT solution_id
            FROM solutions
            WHERE agg_score = (SELECT MAX(agg_score) FROM solutions)
        """)
        result = self.cursor.fetchone()
        if result:
            self.best_solution = id_to_solution[result[0]]

        self.id_to_solution = id_to_solution


    def _touch_solution(self, solution: Solution):
        if not solution.loaded:
            solution.load(self.goo_config, self.cursor)

        if solution.id in self.loaded_cache:
            self.loaded_cache.move_to_end(solution.id)
        else:
            self.loaded_cache[solution.id] = solution

        if len(self.loaded_cache) > self.max_loaded_solutions:
            _, old_solution = self.loaded_cache.popitem(last=False)
            old_solution.unload()

    def load_solution(self, solution: Solution) -> Solution:
        self._touch_solution(solution)
        return solution

    def unload_solution(self, solution: Solution):
        if solution.id in self.loaded_cache:
            del self.loaded_cache[solution.id]
        if solution.loaded:
            solution.unload()

    def add_root(self, solution: Solution):
        self.root_solution = solution
        self.add_solution(solution)

    def add_solution(self, solution: Solution):
        self._touch_solution(solution)

        self.solutions.add(solution, (solution.score, solution.id))
        self.id_to_solution[solution.id] = solution

        solution.save(self.cursor)
        self.conn.commit()

        logger.info(f"added solution {solution.id} (score: {solution.score})")

        if self.best_solution is None or self.best_solution.score < solution.score:
            self.best_solution = solution
            logger.info("new best solution!")

    def clear_database(self):
        self.conn.commit()
        for table in ["solutions", "solutions_tree", "runs"]:
            self.cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()
    
        self.cursor.execute("VACUUM")
        self.conn.commit()
    
        for solution in self.loaded_cache.values():
            if solution.loaded:
                solution.unload()
        self.loaded_cache.clear()
    
        for solution in self.id_to_solution.values():
            solution.clear_refs()
    
        self.id_to_solution.clear()
        self.solutions = SortedList()
    
        self.best_solution = None
        self.root_solution = None
    
        Solution.id_counter = 0

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass

        for solution in self.id_to_solution.values():
            solution.clear_refs()