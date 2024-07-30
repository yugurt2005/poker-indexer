use rand::Rng;
use std::cmp;

const ROUND_SHIFT: usize = 4;
const ROUND_MASK: u64 = 0b1111;

pub struct Indexer {
    count: u64,
    cards_per_round: Vec<u8>,
    permutation_to_config: Vec<u64>,
    config_offset: Vec<u64>,
    configurations: Vec<[u64; 4]>,
    config_suit_size: Vec<[u64; 4]>,
    index_to_rank_set: Vec<Vec<u64>>,
}

pub fn biggest_set_bit_position(n: u64) -> Option<u32> {
    if n == 0 {
        None // No set bits in 0.
    } else {
        Some(64 - n.leading_zeros())
    }
}

impl Indexer {
    pub fn new(cards_per_round: Vec<u8>) -> Indexer {
        let mut indexer = Indexer {
            count: 0,
            cards_per_round,
            permutation_to_config: vec![0; 1e6 as usize],
            config_offset: Vec::new(),
            configurations: Vec::new(),
            config_suit_size: Vec::new(),
            index_to_rank_set: Vec::new(),
        };

        indexer.enumerate_configs(
            0,
            0,
            indexer.cards_per_round[0] as u64,
            [0; 4],
            [0; 4],
            0b1110,
        );
        indexer.enumerate_perms(0, 0, indexer.cards_per_round[0] as u64, [0; 4], [0; 4]);
        indexer.enumerate_index_sets();

        indexer
    }

    pub fn enumerate_configs(
        &mut self,
        suit: usize,
        round: usize,
        remaining: u64,
        used: [u64; 4],
        configuration: [u64; 4],
        equal: u8,
    ) {
        if suit == 4 {
            // if self.count < 25 {
            //     println!("-------------");
            //     println!(
            //         "{} {} {} {}",
            //         configuration[0] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //         configuration[0] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //         configuration[0] >> ROUND_SHIFT & ROUND_MASK,
            //         configuration[0] & ROUND_MASK
            //     );
            //     println!(
            //         "{} {} {} {}",
            //         configuration[1] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //         configuration[1] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //         configuration[1] >> ROUND_SHIFT & ROUND_MASK,
            //         configuration[1] & ROUND_MASK
            //     );
            //     println!(
            //         "{} {} {} {}",
            //         configuration[2] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //         configuration[2] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //         configuration[2] >> ROUND_SHIFT & ROUND_MASK,
            //         configuration[2] & ROUND_MASK
            //     );
            //     println!(
            //         "{} {} {} {}",
            //         configuration[3] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //         configuration[3] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //         configuration[3] >> ROUND_SHIFT & ROUND_MASK,
            //         configuration[3] & ROUND_MASK
            //     );
            //     println!("-------------");
            // }

            self.config_offset.push(1);
            let mut s = 0;
            self.config_suit_size.push([0; 4]);

            while s < 4 {
                let mut size = 1;

                let mut rem = 13;
                for j in 0..=round {
                    let num = configuration[s]
                        >> ROUND_SHIFT * (self.cards_per_round.len() - j - 1)
                        & ROUND_MASK;
                    size *= self.choose(rem, num as u32);
                    // println!("({} {}) {}", rem, num, size);
                    rem -= num as u32;
                }

                if self.count < 10 {
                    // println!("{} {}", s, size);
                }

                let mut j = s + 1;
                while j < 4 && configuration[j] == configuration[s] {
                    j += 1;
                }

                self.config_offset[self.count as usize] *=
                    self.choose(size as u32 + (j - s - 1) as u32, (j - s) as u32);

                self.config_suit_size[self.count as usize][s] = size;

                s = j;
            }

            self.configurations.push(configuration);

            self.count += 1;

            if round < self.cards_per_round.len() - 1 {
                self.enumerate_configs(
                    0,
                    round + 1,
                    self.cards_per_round[round + 1] as u64,
                    used,
                    configuration,
                    equal,
                );
            }
            return;
        }

        let min = if suit == 3 { remaining } else { 0 };
        let mut max = cmp::min(remaining, 13 - used[suit]);

        let mut previous = 14;
        let was_equal = (equal & (1 << suit)) != 0;
        if was_equal {
            previous = configuration[suit - 1]
                >> ROUND_SHIFT * (self.cards_per_round.len() - round - 1)
                & ROUND_MASK;
            if previous < max {
                max = previous;
            }
        }

        // println!("{} {} {} {} {}", suit, round, remaining, min, max);

        for i in min..=max {
            let mut new_configuration = configuration.clone();
            new_configuration[suit] |= i << ROUND_SHIFT * (self.cards_per_round.len() - round - 1);

            let mut new_used = used.clone();
            new_used[suit] += i;

            let mut new_equal = equal;
            if i == previous {
                new_equal |= 1 << suit;
            } else {
                new_equal &= !(1 << suit);
            }
            self.enumerate_configs(
                suit + 1,
                round,
                remaining - i,
                new_used,
                new_configuration,
                new_equal,
            );
        }
    }

    pub fn enumerate_perms(
        &mut self,
        suit: usize,
        round: usize,
        remaining: u64,
        used: [u64; 4],
        perm: [u64; 4],
    ) {
        if suit == 4 {
            // println!("-------------");
            // println!(
            //     "{} {} {} {}",
            //     perm[0] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //     perm[0] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //     perm[0] >> ROUND_SHIFT & ROUND_MASK,
            //     perm[0] & ROUND_MASK
            // );
            // println!(
            //     "{} {} {} {}",
            //     perm[1] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //     perm[1] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //     perm[1] >> ROUND_SHIFT & ROUND_MASK,
            //     perm[1] & ROUND_MASK
            // );
            // println!(
            //     "{} {} {} {}",
            //     perm[2] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //     perm[2] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //     perm[2] >> ROUND_SHIFT & ROUND_MASK,
            //     perm[2] & ROUND_MASK
            // );
            // println!(
            //     "{} {} {} {}",
            //     perm[3] >> ROUND_SHIFT * 3 & ROUND_MASK,
            //     perm[3] >> ROUND_SHIFT * 2 & ROUND_MASK,
            //     perm[3] >> ROUND_SHIFT & ROUND_MASK,
            //     perm[3] & ROUND_MASK
            // );
            // println!("-------------");

            let mut sorted_perm = perm.clone();

            sorted_perm.sort_by(|a, b| b.cmp(a));

            // for s in 0..4 {
            //     println!("{}: {}", s, sorted_perm[s]);
            // }

            let idx = self.permutation(round + 1, sorted_perm);

            // println!("{}", idx);

            let mut lo = 0;
            let mut hi = self.configurations.len() as u64 - 1;
            while lo != hi {
                let mid = (lo + hi) / 2;

                let mut comp = 0;
                for r in 0..self.cards_per_round.len() {
                    for s in 0..4 {
                        let config_count = self.configurations[mid as usize][s]
                            >> (self.cards_per_round.len() - r - 1) * ROUND_SHIFT
                            & ROUND_MASK;
                        let perm_count = sorted_perm[s]
                            >> (self.cards_per_round.len() - r - 1) * ROUND_SHIFT
                            & ROUND_MASK;
                        if config_count != perm_count {
                            if config_count < perm_count {
                                comp = -1;
                            } else {
                                comp = 1;
                            }
                            break;
                        }
                    }
                    if comp != 0 {
                        break;
                    }
                }

                if comp == 0 {
                    lo = mid;
                    hi = mid;
                } else if comp == -1 {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            // if sorted_perm[0] == 8977 {
            // println!("{:?} {} {}", sorted_perm, idx, lo);
            // }

            self.permutation_to_config[idx as usize] = lo;
            // println!("{:?} {} {}", sorted_perm, idx, lo);

            for i in 0..4 {
                // println!(
                //     "{}: {} {}",
                //     i, sorted_perm[i], self.configurations[lo as usize][i]
                // );
                assert!(sorted_perm[i] == self.configurations[lo as usize][i]);
            }

            if round < self.cards_per_round.len() - 1 {
                self.enumerate_perms(
                    0,
                    round + 1,
                    self.cards_per_round[round + 1] as u64,
                    used,
                    perm,
                );
            }

            return;
        }

        let min = if suit == 3 { remaining } else { 0 };
        let max = cmp::min(remaining, 13 - used[suit]);

        for i in min..=max {
            let mut new_perm = perm.clone();
            new_perm[suit] |= i << ROUND_SHIFT * (self.cards_per_round.len() - round - 1);

            let mut new_used = used.clone();
            new_used[suit] += i;

            self.enumerate_perms(suit + 1, round, remaining - i, new_used, new_perm);
        }
    }

    fn choose(&self, n: u32, k: u32) -> u64 {
        if n < k {
            return 0;
        }

        let mut res: u64 = 1;
        for i in 0..k {
            res *= n as u64 - i as u64;
        }
        for i in 0..k {
            res /= i as u64 + 1;
        }
        res
    }

    fn permutation(&self, round: usize, mut cards: [u64; 4]) -> u64 {
        cards.sort_by(|a, b| b.cmp(a));

        let mut index: u64 = 0;
        let mut multiplier: u64 = 1;
        for r in 0..round {
            let mut remaining = self.cards_per_round[r as usize] as u64;
            for card in cards {
                let num = card >> (self.cards_per_round.len() - r - 1) * ROUND_SHIFT & ROUND_MASK;
                index += multiplier * num;
                multiplier *= remaining + 1;
                remaining -= num as u64;
            }
        }

        index
    }

    pub fn enumerate_index_sets(&mut self) {
        let mut res = vec![vec![0; 1 << 13]; 14];
        for i in 0..1 << 13 {
            res[(i as u32).count_ones() as usize][self.index_set(i, 0) as usize] = i;
        }
        self.index_to_rank_set = res;
    }

    pub fn index_set(&self, mut x: u64, used: u64) -> u64 {
        let m = x.count_ones();

        let mut res = 0;
        for i in 0..m {
            let b = biggest_set_bit_position(x).unwrap();
            let y = 1 << (b - 1);
            x ^= y;

            let rank = b - ((y - 1) & used).count_ones();

            res += self.choose(rank - 1, m - i);

            // println!("{} {} {} {}", res, b, rank - 1, m - i);
        }

        return res;
    }

    pub fn index_group(&self, mut sets: Vec<u64>, used: u64) -> u64 {
        if sets.len() == 0 {
            return 0;
        }

        let x = sets.remove(0);

        let next = self.index_group(sets, used | x);

        let mul = self.choose(13 - used.count_ones(), x.count_ones());
        // println!("{} {}", mul, next);
        let mut idx = mul * next;

        idx += self.index_set(x, used);

        return idx;
    }

    pub fn multiset_colex(&self, multisets: Vec<u64>) -> u64 {
        let m = multisets.len() as u64;

        let mut idx = 0;
        for i in 0..m {
            idx += self.choose((multisets[i as usize] + m - i - 1) as u32, (m - i) as u32);
        }
        idx
    }

    fn get_config(&self, perm_index: u64) -> u64 {
        self.permutation_to_config[perm_index as usize]
    }

    fn get_offset(&self, config_index: u64) -> u64 {
        let mut res = 0;
        for i in 0..config_index {
            let mut num = 0;
            for j in 0..4 {
                for k in 0..4 {
                    num += self.configurations[i as usize][j]
                        >> (self.cards_per_round.len() - k - 1) * ROUND_SHIFT
                        & ROUND_MASK;
                }
            }

            // println!("{}: {:?} | {}", i, self.configurations[i as usize], num);

            if num == 7 {
                res += self.config_offset[i as usize]
            }
        }

        res
    }

    pub fn index(&self, sets_by_round: Vec<u64>) -> u64 {
        let mut sets_by_suit = vec![Vec::new(); 4];

        for i in 0..self.cards_per_round.len() {
            for j in 0..4 {
                sets_by_suit[j].push(sets_by_round[i] >> (13 * j) & ((1 << 13) - 1));
            }
        }

        let mut idxs = Vec::new();
        for sets in sets_by_suit {
            let lexo_order: Vec<u8> = sets.iter().map(|set| set.count_ones() as u8).collect();
            idxs.push((lexo_order, self.index_group(sets, 0)));
        }

        // println!(
        //     "group indices {:?}",
        //     idxs.iter().map(|x| x.1).collect::<Vec<u64>>()
        // );

        idxs.sort_by(|a, b| b.cmp(a));

        let mut idx = 0;
        let mut i = 0;
        let mut cur = 1;
        while i < idxs.len() {
            let mut j = i + 1;
            while j < idxs.len() && idxs[j].0 == idxs[i].0 {
                j += 1;
            }

            let cnt = j - i;

            // println!("{} {} {}", i, j, cnt);

            let value = self.multiset_colex((i..j).map(|x| idxs[x].1).collect::<Vec<u64>>());
            idx += cur * value;

            let mut num: u32 = 13;
            let mut size: u64 = 1;
            for x in idxs[i].0.iter() {
                size *= self.choose(num, *x as u32);
                num -= *x as u32;
            }

            cur *= self.choose(size as u32 + (cnt as u32) - 1, cnt as u32);

            i = j;
        }

        let mut permut = [0; 4];
        for s in 0..4 {
            for i in 0..self.cards_per_round.len() {
                permut[s] |=
                    (idxs[s].0[i] as u64) << (self.cards_per_round.len() - i - 1) * ROUND_SHIFT;
            }
        }

        let perm_idx = self.permutation(self.cards_per_round.len(), permut);
        let config_idx = self.get_config(perm_idx);
        let config_offset = self.get_offset(config_idx);
        // println!(
        //     "{} {} {:?} | {}",
        //     config_idx, perm_idx, permut, config_offset
        // );

        // println!(
        //     "idx before offset = {}; answer = {}",
        //     idx,
        //     idx + config_offset
        // );

        return idx + config_offset;
    }

    pub fn unindex_group(&self, mut group_idx: u64, configuration: u64) -> Vec<u64> {
        let mut m = 0;
        let mut used = 0;
        let mut answer = Vec::new();

        // println!("configuration {}", configuration);

        for r in 0..self.cards_per_round.len() {
            let n = configuration >> (self.cards_per_round.len() - r - 1) * ROUND_SHIFT
                & ROUND_MASK as u64;
            let round_size = self.choose(13 - m as u32, n as u32);
            m += n;

            let round_idx = group_idx % round_size;
            group_idx /= round_size;

            // println!("{} {}: {} {} {}", r, n, round_idx, round_size, group_idx);

            let mut input = self.index_to_rank_set[n as usize][round_idx as usize];

            let mut card = 0;
            for _k in 0..n {
                let mut shifted_card = ((input as i32) & -(input as i32)) as u64;
                input ^= shifted_card;

                let mut j = (((shifted_card - 1) << 1 | 1) & used).count_ones();
                while j > 0 {
                    shifted_card <<= 1;
                    j -= 1;
                    if shifted_card & used > 0 {
                        j += 1;
                    }
                }
                card |= shifted_card;
            }

            answer.push(card);

            used |= card;
        }

        answer
    }

    pub fn unindex(&self, mut idx: u64) -> Vec<u64> {
        println!("{}", idx);

        let mut lo = 0;
        let mut hi = self.configurations.len() - 1;
        while lo != hi {
            let mid = (lo + hi + 1) / 2;

            if self.get_offset(mid as u64) <= idx {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        idx = idx - self.get_offset(lo as u64);

        // println!(
        //     "{} {}: {:?} = new index {}",
        //     lo,
        //     self.get_offset(lo as u64),
        //     self.configurations[lo],
        //     idx
        // );

        let mut group_indices = Vec::new();

        let mut s = 0;
        while s < 4 {
            let mut j = s + 1;
            while j < 4 && self.configurations[lo][j] == self.configurations[lo][s] {
                // println!(
                //     "{} {}: {} {}",
                //     s, j, self.configurations[lo][j], self.configurations[lo][s]
                // );
                j += 1;
            }

            let size = self.config_suit_size[lo][s];
            let group_size = self.choose(size as u32 + (j - s - 1) as u32, (j - s) as u32);
            let mut group_index = idx % group_size;
            idx /= group_size;

            // println!(
            //     "{} - {} == {} {} {} {}",
            //     s, j, size, group_size, group_index, idx
            // );

            while s < j - 1 {
                let mut lo = 0;
                let mut hi = size as usize;

                while lo != hi {
                    let mid = (lo + hi + 1) / 2;

                    if self.choose((mid as u32) + (j - s - 1) as u32, (j - s) as u32) <= group_index
                    {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }

                group_indices.push(lo as u64);

                group_index -= self.choose((lo as u32) + (j - s - 1) as u32, (j - s) as u32);

                s += 1
            }

            group_indices.push(group_index);
            s += 1
        }

        // println!("group indices {:?}", group_indices);

        let mut answer = Vec::new();
        for s in 0..4 {
            answer.push(self.unindex_group(group_indices[s], self.configurations[lo][s]))
        }

        let mut answer2 = vec![0; self.cards_per_round.len() as usize];
        for i in 0..self.cards_per_round.len() {
            for s in 0..4 {
                answer2[i] |= answer[s][i] << 13 * s;
            }
        }

        answer2
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    // fn test_enumerate_configs() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     println!("{}", indexer.count);

    //     let mut sum = 0;
    //     for i in 0..indexer.count {
    //         println!("{}", indexer.config_offset[i as usize]);
    //         sum += indexer.config_offset[i as usize];
    //     }

    //     println!("Sum {}", sum);
    // }

    // #[test]
    // fn test_enumerate_perms() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);

    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);
    // }

    // #[test]
    // fn test_index() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);

    //     let sets_by_suit = vec![
    //         vec![1 << 12, 1, 2, 4],
    //         vec![1 << 12, 3, 0, 0],
    //         vec![0, 0, 0, 0],
    //         vec![0, 0, 0, 0],
    //     ];

    //     let idx = indexer.index(sets_by_suit);

    //     println!("{}", idx);
    // }

    // #[test]
    // fn test_correctness_smallest() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);

    //     let sets_by_suit = vec![
    //         vec![1, 0, 0, 0],
    //         vec![1, 0, 0, 0],
    //         vec![0, 3, 0, 0],
    //         vec![0, 1, 2, 8],
    //     ];

    //     // println!("{:?}", indexer.configurations[0]);

    //     let mut cards = vec![0; 7];

    //     let mut pos = 0;
    //     for i in 0..4 {
    //         for j in 0..4 {
    //             for k in 0..13 {
    //                 if sets_by_suit[j][i] >> k & 1 != 0 {
    //                     cards[pos] = k << 2 | (j as u8);
    //                     pos += 1;
    //                 }
    //             }
    //         }
    //     }

    //     let idx = indexer.index(sets_by_suit);

    //     let correct_indexer = hand_indexer::Indexer::new(vec![2, 3, 1, 1]).unwrap();

    //     let actual = *correct_indexer.index_all(&cards).unwrap().last().unwrap();

    //     println!("{} {}", idx, actual);
    //     assert!(actual as u64 == idx);
    // }

    // #[test]
    // fn test_correctness_biggest() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);

    //     // println!("{:?}", indexer.get_offset(indexer.config_offset.len() as u64 - 1));

    //     let sets_by_suit = vec![
    //         vec![3, 4 + 8 + 16, 32, 128],
    //         vec![0, 0, 0, 0],
    //         vec![0, 0, 0, 0],
    //         vec![0, 0, 0, 0],
    //     ];

    //     let mut cards = vec![0; 7];

    //     let mut pos = 0;
    //     for i in 0..4 {
    //         for j in 0..4 {
    //             for k in 0..13 {
    //                 if sets_by_suit[j][i] >> k & 1 != 0 {
    //                     cards[pos] = k << 2 | (j as u8);
    //                     pos += 1;
    //                 }
    //             }
    //         }
    //     }

    //     let idx = indexer.index(sets_by_suit);

    //     let correct_indexer = hand_indexer::Indexer::new(vec![2, 3, 1, 1]).unwrap();

    //     let actual = *correct_indexer.index_all(&cards).unwrap().last().unwrap();

    //     println!("{} {}", idx, actual);
    //     assert!(actual as u64 == idx);
    // }

    // #[test]
    // fn test_correctness_random() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);

    //     // println!("{:?}", indexer.get_offset(indexer.config_offset.len() as u64 - 1));

    //     // let sets_by_suit = vec![
    //     //     vec![3, 0, 0, 4],
    //     //     vec![0, 3, 0, 0],
    //     //     vec![0, 1, 2, 0],
    //     //     vec![0, 0, 0, 0],
    //     // ];

    //     let sets_by_suit = vec![
    //         // vec![1 << 11 | 1 << 12, 0, 0, 4],
    //         // vec![0, 1 << 4 | 1 << 11, 0, 0],
    //         // vec![0, 1 << 5, 1 << 6, 0],
    //         // vec![0, 0, 0, 0],
    //         vec![512, 1024, 256, 0],
    //         vec![256, 1, 0, 0],
    //         vec![0, 16, 0, 0],
    //         vec![0, 0, 0, 8],
    //         // vec![1, 2, 4, 0],
    //         // vec![1, 2, 0, 0],
    //         // vec![0, 1, 0, 0],
    //         // vec![0, 0, 0, 1],
    //     ];

    //     let mut cards = vec![0; 7];

    //     let mut pos = 0;
    //     for i in 0..4 {
    //         for j in 0..4 {
    //             for k in 0..13 {
    //                 if sets_by_suit[j][i] >> k & 1 != 0 {
    //                     cards[pos] = k << 2 | (j as u8);
    //                     pos += 1;
    //                 }
    //             }
    //         }
    //     }

    //     let idx = indexer.index(sets_by_suit);

    //     let correct_indexer = hand_indexer::Indexer::new(vec![2, 3, 1, 1]).unwrap();

    //     let actual = *correct_indexer.index_all(&cards).unwrap().last().unwrap();

    //     // println!("{} {}", idx - 1983459036, actual - 2012883288);
    //     println!("{} {}", idx - 1143448592, actual - 1105748072);
    // }

    // #[test]
    // fn test_unindex() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);
    //     indexer.enumerate_index_sets();

    //     // println!("{:?}", indexer.get_offset(indexer.config_offset.len() as u64 - 1));

    //     let sets_by_suit = vec![
    //         vec![1 << 5, 1 << 2 | 1 << 10, 0, 0],
    //         vec![1 << 2, 0, 0, 0],
    //         vec![0, 1 << 9, 0, 0],
    //         vec![0, 0, 1, 2],
    //     ];

    //     let mut cards = vec![0; 7];

    //     let mut pos = 0;
    //     for i in 0..4 {
    //         for j in 0..4 {
    //             for k in 0..13 {
    //                 if sets_by_suit[j][i] >> k & 1 != 0 {
    //                     cards[pos] = k << 2 | (j as u8);
    //                     pos += 1;
    //                 }
    //             }
    //         }
    //     }

    //     let idx = indexer.index(sets_by_suit);

    //     indexer.unindex(idx);
    // }

    // #[test]
    // fn test_unindex_full() {
    //     let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

    //     indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
    //     indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);
    //     indexer.enumerate_index_sets();

    //     // println!("{:?}", indexer.get_offset(indexer.config_offset.len() as u64 - 1));

    //     let sets_by_suit = vec![
    //         vec![1 << 5, 1 << 2 | 1 << 10, 0, 0],
    //         vec![1 << 2, 0, 0, 0],
    //         vec![0, 1 << 9, 0, 0],
    //         vec![0, 0, 1, 2],
    //     ];

    //     let sbs2 = sets_by_suit.clone();

    //     let idx = indexer.index(sets_by_suit);

    //     println!("UNINDEXING");

    //     let sbs = indexer.unindex(idx);

    //     println!("{:?}", sbs);
    //     println!("{:?}", sbs2);
    // }

    #[test]
    fn test_unindex_bidirectional() {
        let mut indexer = Indexer::new(vec![2, 3, 1, 1]);

        indexer.enumerate_configs(0, 0, 2, [0; 4], [0; 4], 0b1110);
        indexer.enumerate_perms(0, 0, 2, [0; 4], [0; 4]);
        indexer.enumerate_index_sets();

        // println!("{:?}", indexer.get_offset(indexer.config_offset.len() as u64 - 1));

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let index = rng.gen_range(0..(2e9 as u64));

            let res = indexer.index(indexer.unindex(index));

            println!("test inputs {} {}", index, res);
            assert!(res == index);
        }

        // let index = 1154974622;
        // let unindexed = indexer.unindex(index);
        // println!("{:?}", unindexed);
        // println!("INDEXING");
        // let res = indexer.index(unindexed);

        // // println!("UNINDEXING");
        // // let unindexed2 = indexer.unindex(res);
        // // println!("{:?}", unindexed2);

        // println!("test inputs {} {}", index, res);
        // assert!(res == index);
    }
}

// pub fn biggest_set_bit_position(n: u64) -> Option<u32> {
//     if n == 0 {
//         None // No set bits in 0.
//     } else {
//         Some(64 - n.leading_zeros())
//     }
// }

// pub fn index_set(mut x: u64, used: u64) -> u64 {
//     let m = x.count_ones();

//     let mut res = 0;
//     for i in 0..m {
//         let b = biggest_set_bit_position(x).unwrap();
//         let y = 1 << (b - 1);
//         x ^= y;

//         let rank = b - ((y - 1) & used).count_ones();

//         res += choose(rank - 1, m - i);

//         // println!("{} {} {} {}", res, b, rank - 1, m - i);
//     }

//     return res;
// }

// pub fn index_group(mut sets: Vec<u64>, used: u64) -> u64 {
//     if sets.len() == 0 {
//         return 0;
//     }

//     let x = sets.remove(0);

//     let next = index_group(sets, used | x);

//     let mul = choose(13 - used.count_ones(), x.count_ones());
//     // println!("{} {}", mul, next);
//     let mut idx = mul * next;

//     idx += index_set(x, used);

//     return idx;
// }

// pub fn multiset_colex(multisets: Vec<u64>) -> u64 {
//     let m = multisets.len() as u64;

//     let mut idx = 0;
//     for i in 0..m {
//         idx += choose((multisets[i as usize] + m - i - 1) as u32, (m - i) as u32);
//     }
//     idx
// }

// pub fn index_hand(sets_by_suit: Vec<Vec<u64>>) -> u64 {
//     let mut idxs = Vec::new();
//     for sets in sets_by_suit {
//         let lexo_order: Vec<u8> = sets.iter().map(|set| set.count_ones() as u8).collect();
//         idxs.push((lexo_order, index_group(sets, 0)));
//     }

//     idxs.sort_by(|a, b| b.cmp(a));

//     let mut idx = 0;
//     let mut i = 0;
//     let mut cur = 1;
//     while i < idxs.len() {
//         if idxs[i].0[0] == 0 && idxs[i].0[1] == 0 {
//             break;
//         }

//         let mut j = i + 1;
//         while j < idxs.len() && idxs[j].0 == idxs[i].0 {
//             j += 1;
//         }

//         let cnt = j - i;

//         let value = multiset_colex((i..j).map(|x| idxs[x].1).collect::<Vec<u64>>());
//         idx += cur * value;

//         let mut num: u32 = 13;
//         let mut size: u64 = 1;
//         for x in idxs[i].0.iter() {
//             size *= choose(num, *x as u32);
//             num -= *x as u32;
//         }

//         println!("{}", size);

//         cur *= choose(size as u32 + (cnt as u32) - 1, cnt as u32);

//         i = j;
//     }

//     return idx;
// }
