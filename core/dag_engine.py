"""Directed Acyclic Graph engine for managing test case dependencies."""

from typing import Dict, List, Optional, Set

import networkx as nx

from core.models import TestCase


class DependencyCycleError(Exception):
    """Raised when a cycle is detected in the dependency graph."""

    pass


class TestSuiteDAG:
    """Manages test case dependencies as a Directed Acyclic Graph."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._node_map: Dict[str, TestCase] = {}

    def build(self, test_cases: List[TestCase]) -> "TestSuiteDAG":
        """Build the DAG from a list of test cases.

        Args:
            test_cases: List of TestCase objects with dependency information.

        Returns:
            Self for method chaining.

        Raises:
            DependencyCycleError: If the dependency graph contains a cycle.
        """
        self.graph.clear()
        self._node_map.clear()

        # Add all nodes first
        for tc in test_cases:
            self.graph.add_node(tc.id, test_case=tc)
            self._node_map[tc.id] = tc

        # Add edges for dependencies
        for tc in test_cases:
            for dep_id in tc.dependencies:
                if dep_id in self._node_map:
                    self.graph.add_edge(dep_id, tc.id)
                else:
                    # Log warning about missing dependency
                    print(f"Warning: Test case {tc.id} depends on unknown test case {dep_id}")

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise DependencyCycleError(
                f"Dependency cycle detected: {cycles}"
            )

        return self

    def get_topological_order(self) -> List[str]:
        """Get test case IDs in topological order.

        Returns:
            List of test case IDs ordered by dependencies.
        """
        return list(nx.topological_sort(self.graph))

    def get_ready_nodes(self, completed: Set[str]) -> List[str]:
        """Get nodes whose dependencies are all satisfied.

        Args:
            completed: Set of test case IDs that have completed.

        Returns:
            List of test case IDs ready for execution.
        """
        ready = []
        for node in self.graph.nodes():
            if node in completed:
                continue
            predecessors = set(self.graph.predecessors(node))
            if predecessors.issubset(completed):
                ready.append(node)
        return ready

    def get_dependents(self, tc_id: str) -> List[str]:
        """Get all test cases that directly or indirectly depend on the given test case.

        Args:
            tc_id: Test case ID.

        Returns:
            List of dependent test case IDs.
        """
        if tc_id not in self.graph:
            return []
        # Get all descendants (both direct and transitive)
        descendants = nx.descendants(self.graph, tc_id)
        return list(descendants)

    def skip_subtree(self, failed_tc_id: str) -> List[str]:
        """Mark a test case and all its dependents as skipped.

        Uses DFS to recursively mark all children.

        Args:
            failed_tc_id: ID of the failed test case.

        Returns:
            List of test case IDs that were marked as skipped.
        """
        from core.enums import TestStatus

        skipped: List[str] = []

        def _mark_skipped(node_id: str) -> None:
            if node_id not in self._node_map:
                return
            tc = self._node_map[node_id]
            if tc.status == TestStatus.PENDING:
                tc.status = TestStatus.SKIPPED
                skipped.append(node_id)
            # Recursively mark all dependents
            for dependent in self.graph.successors(node_id):
                _mark_skipped(dependent)

        _mark_skipped(failed_tc_id)
        return skipped

    def is_blocked(self, tc_id: str, failed_nodes: Set[str]) -> bool:
        """Check if a test case is blocked by any failed dependency.

        Args:
            tc_id: Test case ID to check.
            failed_nodes: Set of test case IDs that have failed.

        Returns:
            True if any dependency (direct or transitive) has failed.
        """
        if tc_id not in self.graph:
            return False
        ancestors = nx.ancestors(self.graph, tc_id)
        return not ancestors.isdisjoint(failed_nodes)

    def get_execution_waves(self) -> List[List[str]]:
        """Group test cases into waves for parallel execution.

        Each wave contains test cases whose dependencies are all in previous waves.

        Returns:
            List of waves, where each wave is a list of test case IDs.
        """
        waves: List[List[str]] = []
        executed: Set[str] = set()

        while len(executed) < self.graph.number_of_nodes():
            wave = [
                node for node in self.graph.nodes()
                if node not in executed
                and all(pred in executed for pred in self.graph.predecessors(node))
            ]
            if not wave:
                break
            waves.append(wave)
            executed.update(wave)

        return waves

    def get_test_case(self, tc_id: str) -> Optional[TestCase]:
        """Retrieve a test case by ID.

        Args:
            tc_id: Test case ID.

        Returns:
            TestCase object or None if not found.
        """
        return self._node_map.get(tc_id)

    def __len__(self) -> int:
        """Return the number of test cases in the DAG."""
        return self.graph.number_of_nodes()

    def __contains__(self, tc_id: str) -> bool:
        """Check if a test case ID exists in the DAG."""
        return tc_id in self._node_map
