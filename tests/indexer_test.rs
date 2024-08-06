use rand::Rng;

use poker_indexer::Indexer;

#[test]
fn stress_test_bidirectional_random() {
    let indexer = Indexer::new(vec![2, 3, 1, 1]);

    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let input = rng.gen_range(0..indexer.count[3]);

        let actual = indexer.index(indexer.unindex(input, 3));
        let expect = input;

        assert_eq!(actual, expect);
    }
}

#[test]
fn stress_test_bidirectional_all() {
    let indexer = Indexer::new(vec![2, 3]);

    for i in 0..indexer.count[1] as u32 {
        let actual = indexer.index(indexer.unindex(i, 1));
        let expect = i;

        assert_eq!(actual, expect);
    }
}