
pub struct Indexer {
    indexer: hand_indexer::Indexer,
}

impl Indexer {
    pub fn new(rounds: Vec<usize>) -> Self {
        let indexer = hand_indexer::Indexer::new(rounds).unwrap();
        Indexer { indexer }
    }

    pub fn index(&self, cards: Vec<u64>) -> u64 {
        let mut input: Vec<u8> = Vec::new();
        for mut card in cards {
            while card > 0 {
                let i = card.trailing_zeros();

                input.push(((i % 13) << 2 | (i / 13)) as u8);

                card &= card - 1;
            }
        }

        let res = self.indexer.index_all(&input[..]).unwrap();

        res[res.len() - 1] as u64
    }

    pub fn unindex(&self, index: u64) -> Vec<u64> {
        let res = self.indexer.unindex(index as usize, 0).unwrap();

        let mut output = 0;
        for i in res {
            output |= 1 << ((i >> 2) as u64) << (13 * (i & 3));
        }

        vec![output]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexer() {
        let indexer = Indexer::new(vec![2]);
        let cards = vec![1 << 11 | 1 << 12];
        let index = indexer.index(cards.clone());
        let unindexed = indexer.unindex(index);

        assert_eq!(index, 168);
        assert_eq!(unindexed, cards);
    }
}