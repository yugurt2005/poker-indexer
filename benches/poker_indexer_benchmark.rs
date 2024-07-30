use criterion::{black_box, criterion_group, criterion_main, Criterion};

use hand_indexer;
use poker_indexer::Indexer;

fn poker_indexer_benchmark(c: &mut Criterion) {
    // let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    // indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    // indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);
    // indexer.enumerate_index_sets();

    // c.bench_function("Index", |b| {
    //     b.iter(|| {
    //         let sets_by_suit = vec![
    //             vec![1 << 5, 1 << 2 | 1 << 10, 0, 0],
    //             vec![1 << 2, 0, 0, 0],
    //             vec![0, 1 << 9, 0, 0],
    //             vec![0, 0, 1, 2],
    //         ];

    //         indexer.index(sets_by_suit);
    //     })
    // });

    let sets_by_suit = vec![
        // vec![1 << 11 | 1 << 12, 0, 0, 4],
        // vec![0, 1 << 4 | 1 << 11, 0, 0],
        // vec![0, 1 << 5, 1 << 6, 0],
        // vec![0, 0, 0, 0],
        vec![512, 1024, 256, 0],
        vec![256, 1, 0, 0],
        vec![0, 16, 0, 0],
        vec![0, 0, 0, 8],
        // vec![1, 2, 4, 0],
        // vec![1, 2, 0, 0],
        // vec![0, 1, 0, 0],
        // vec![0, 0, 0, 1],
    ];

    let mut cards = vec![0; 7];

    let mut pos = 0;
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..13 {
                if sets_by_suit[j][i] >> k & 1 != 0 {
                    cards[pos] = k << 2 | (j as u8);
                    pos += 1;
                }
            }
        }
    }

    let indexer = hand_indexer::Indexer::new(vec![2, 3, 1, 1]).unwrap();

    c.bench_function("Index", |b| {
        b.iter(|| {
            indexer.index_all(black_box(&cards));
        })
    });
}

criterion_group!(benches, poker_indexer_benchmark);
criterion_main!(benches);
