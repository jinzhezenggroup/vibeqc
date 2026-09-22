import itertools
import typing

import pytest
from vibeqc_compiler.common.runtime_domain import (
    RuntimeTaskDomain,
    RuntimeTaskPage,
)


def test_rectangular_domain_matches_product_and_bounded_pages() -> None:
    domain = RuntimeTaskDomain.rectangular((2, 3))
    expected = tuple(itertools.product(range(2), range(3)))

    assert tuple(domain) == expected
    assert domain.logical_size == len(expected)
    assert all(domain.contains(row) for row in expected)
    assert not domain.contains((2, 0))

    pages = tuple(domain.pages(4))
    assert [page.coordinates for page in pages] == [expected[:4], expected[4:]]
    assert [page.offset for page in pages] == [0, 4]
    assert [page.count for page in pages] == [4, 2]
    assert [page.padding for page in pages] == [0, 2]
    assert domain.page_count(4) == 2


def test_nonincreasing_domain_preserves_triples_order_and_closed_form_size() -> None:
    domain = RuntimeTaskDomain.nonincreasing(
        5,
        3,
        outer_start=1,
        outer_stop=3,
    )
    # Build the independent nested-loop oracle explicitly; do not duplicate
    # RuntimeTaskDomain's combinatorial size formula.
    oracle = tuple(
        (a, b, c) for a in range(1, 3) for b in range(a + 1) for c in range(b + 1)
    )
    assert tuple(domain) == oracle
    assert domain.logical_size == len(oracle) == 9
    assert domain.contains((2, 1, 1))
    assert not domain.contains((0, 0, 0))
    assert not domain.contains((2, 1, 2))


def test_runtime_domain_identity_is_semantic_and_page_identity_is_stable() -> None:
    first = RuntimeTaskDomain.nonincreasing(8, 3, outer_start=2, outer_stop=5)
    same = RuntimeTaskDomain.nonincreasing(8, 3, outer_start=2, outer_stop=5)
    different = RuntimeTaskDomain.nonincreasing(8, 3, outer_start=3, outer_stop=5)

    assert first.identity == same.identity
    assert first.identity != different.identity
    first_pages = tuple(first.pages(7))
    same_pages = tuple(same.pages(7))
    assert [page.identity for page in first_pages] == [
        page.identity for page in same_pages
    ]
    assert len({page.identity for page in first_pages}) == len(first_pages)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RuntimeTaskDomain.rectangular(()),
        lambda: RuntimeTaskDomain.rectangular((2, 0)),
        lambda: RuntimeTaskDomain("unknown", (2,)),
        lambda: RuntimeTaskDomain("rectangular", (2,), (0, 1)),
        lambda: RuntimeTaskDomain.nonincreasing(3, 2, outer_start=2, outer_stop=2),
    ],
)
def test_runtime_domain_rejects_invalid_shapes(
    factory: typing.Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


def test_runtime_page_rejects_unbounded_or_malformed_coordinates() -> None:
    domain = RuntimeTaskDomain.rectangular((2, 2))
    with pytest.raises(ValueError):
        RuntimeTaskPage(domain.identity, 2, 0, 0, 1, ())
    with pytest.raises(ValueError):
        RuntimeTaskPage(domain.identity, 2, 0, 0, 1, ((0, 0), (0, 1)))
    with pytest.raises(ValueError):
        RuntimeTaskPage(domain.identity, 2, 0, 0, 2, ((0,),))
    with pytest.raises(ValueError):
        tuple(domain.pages(0))
