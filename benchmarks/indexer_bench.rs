use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{self, Rng};

use poker_indexer::Indexer;
use smallvec::SmallVec;

fn bench_index(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let rounds = vec![2, 3, 1, 1];

    let indexer = Indexer::new(rounds.clone());
    c.bench_function("Indexing", |b| {
        b.iter_batched(
            || {
                let mut input = SmallVec::<[u64; 4]>::new();

                let mut used = 0u64;
                for &round in &rounds {
                    let mut next = 0;
                    for _ in 0..round {
                        loop {
                            let card = 1 << rng.gen_range(0..52);

                            if card & used == 0 {
                                next |= card;
                                used |= card;
                                break;
                            }
                        }
                    }
                    input.push(next);
                }

                input
            },
            |input| {
                indexer.index(black_box(input));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_unindex(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let rounds = vec![2, 3, 1, 1];

    let indexer = Indexer::new(rounds.clone());
    c.bench_function("Unindexing", |b| {
        b.iter_batched(
            || rng.gen_range(0..indexer.count),
            |input| {
                indexer.unindex(black_box(input));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_index, bench_unindex);
criterion_main!(benches);
