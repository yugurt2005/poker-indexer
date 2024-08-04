use std::cmp;

use itertools::Itertools;
use smallvec::{smallvec, SmallVec};

const CHOOSE_SIZE: usize = 15;

const RANKS: u32 = 13;
const SUITS: u32 = 4;

pub struct Indexer {
    rounds: Vec<u32>,

    choose: Vec<Vec<u32>>,

    colex: Vec<u32>,
    index: Vec<Vec<u32>>,
    unset: Vec<Vec<u32>>,

    configs: Vec<[u32; SUITS as usize]>,
    offsets: Vec<u32>,

    sizes: Vec<[u32; SUITS as usize]>,

    map: Vec<usize>,

    order: Vec<[usize; SUITS as usize]>,
}

impl Indexer {
    pub fn new(rounds: Vec<u32>) -> Indexer {
        let count = rounds
            .iter()
            .map(|&x| (x + 1).pow(SUITS) as usize)
            .product();

        let mut indexer = Indexer {
            rounds,

            choose: Vec::new(),

            colex: Vec::new(),
            index: Vec::new(),
            unset: Vec::new(),

            configs: Vec::new(),
            offsets: Vec::new(),

            sizes: Vec::new(),

            map: vec![0; count],

            order: vec![[0; 4]; count],
        };

        indexer.create_helpers();
        indexer.enumerate_configs([0; 4], 0);
        indexer.calculate_configs();
        indexer.build_map([0; 4], 0);

        indexer
    }

    fn create_helpers(&mut self) {
        self.choose = Vec::new();

        for n in 0..CHOOSE_SIZE {
            self.choose.push(vec![1; n + 1]);
            for k in 1..n {
                self.choose[n][k] = self.choose[n - 1][k - 1] + self.choose[n - 1][k];
            }
        }

        self.colex = vec![0; 1 << RANKS];
        for i in 0u32..1 << RANKS {
            let mut x = i;
            let mut c = 1;
            while x > 0 {
                self.colex[i as usize] += self.nck(x.trailing_zeros(), c);

                x &= x - 1;
                c += 1;
            }
        }

        self.index = vec![vec![0; 1 << RANKS]; RANKS as usize + 1];
        for i in 0u32..1 << RANKS {
            self.index[i.count_ones() as usize][self.colex[i as usize] as usize] = i;
        }

        self.unset = vec![vec![0; 1 << RANKS]; RANKS as usize + 1];
        for u in 0..1 << RANKS {
            let mut s = (!u & (1 << RANKS) - 1) as i32;
            let mut i = 0;

            while i < RANKS {
                self.unset[i as usize][u as usize] = s.trailing_zeros();

                s &= s - 1;
                i += 1;
            }
        }
    }

    fn enumerate_configs(&mut self, mut config: [u32; SUITS as usize], round: usize) {
        if round == self.rounds.len() {
            return;
        }

        let n = self.rounds[round] + 1;
        let p = self.rounds.len() - round - 1;

        for i in 0..n.pow(SUITS) {
            let mut x = i;
            let mut s = 0;
            for j in 1..=SUITS {
                config[(SUITS - j) as usize] |= (x % n) << (p * 4);

                s += x % n;
                x /= n;
            }

            if s == self.rounds[round] {
                if (1..SUITS).all(|i| config[i as usize] <= config[i as usize - 1]) {
                    self.configs.push(config);
                }
                self.enumerate_configs(config, round + 1);
            }

            for j in 0..SUITS {
                config[j as usize] >>= p * 4 + 4;
                config[j as usize] <<= p * 4 + 4;
            }
        }

        if round == 0 {
            self.configs.sort();
        }
    }

    fn calculate_configs(&mut self) {
        for config in &self.configs {
            self.sizes.push([0; SUITS as usize]);

            for (key, chunk) in &(0..SUITS as usize).chunk_by(|&x| config[x]) {
                let mut r = RANKS;
                let mut s = 1;
                for k in 0..self.rounds.len() {
                    let c = key >> (self.rounds.len() - k - 1) * 4 & 0b1111;

                    s *= self.nck(r, c);
                    r -= c;
                }

                for x in chunk {
                    self.sizes.last_mut().unwrap()[x] = s;
                }
            }

            println!("{:?}: {:?}", config, self.sizes.last().unwrap());
        }

        self.offsets.push(0);
        for c in 0..self.configs.len() {
            let mut offset = 1;

            let mut cnt = 1;
            let mut pre = 0;

            for i in 1..SUITS as usize {
                if self.configs[c][i] == self.configs[c][pre] {
                    cnt += 1;
                } else {
                    offset = offset * self.nck(self.sizes[c][pre] + cnt - 1, cnt);

                    cnt = 1;
                    pre = i;
                }
            }
            offset *= self.nck(self.sizes[c][pre] + cnt - 1, cnt);

            self.offsets.push(offset + self.offsets[c]);
        }
    }

    fn build_map(&mut self, mut permutation: [u32; SUITS as usize], round: usize) {
        if round == self.rounds.len() {
            return;
        }

        let n = self.rounds[round] + 1;
        let p = self.rounds.len() - round - 1;

        for i in 0..n.pow(SUITS) {
            let mut x = i;
            let mut s = 0;
            for j in 1..=SUITS {
                permutation[(SUITS - j) as usize] |= (x % n) << (p * 4);

                s += x % n;
                x /= n;
            }

            if s == self.rounds[round] {
                let mut p = 0;
                let mut m = 1;
                for i in 0..=round {
                    let mut r = self.rounds[i];
                    for x in permutation {
                        let c = x >> (self.rounds.len() - i - 1) * 4 & 0b1111;

                        p += c * m;
                        m *= r + 1;

                        r -= c;
                    }
                }

                for i in 0..SUITS as usize {
                    for k in 0..SUITS as usize {
                        self.order[p as usize][i] += (permutation[i] >= permutation[k]) as usize;
                    }
                }

                let mut sorted = permutation.clone();

                sorted.sort();
                sorted.reverse();

                self.map[p as usize] = self.configs.binary_search(&sorted).unwrap();

                self.build_map(permutation, round + 1);
            }

            for j in 0..SUITS {
                permutation[j as usize] >>= p * 4 + 4;
                permutation[j as usize] <<= p * 4 + 4;
            }
        }
    }

    fn nck(&self, n: u32, k: u32) -> u32 {
        if n < k {
            return 0;
        }

        if (n as usize) < CHOOSE_SIZE {
            self.choose[n as usize][k as usize]
        } else {
            let mut answer = 1;

            for i in 0..cmp::min(k, n - k) {
                answer *= (n - i) as u64;
                answer /= (i + 1) as u64;
            }

            answer as u32
        }
    }

    fn permutation(&self, input: &SmallVec<[u64; 4]>) -> usize {
        let mut p = 0;
        let mut m = 1;
        for i in 0..input.len() {
            let mut r = self.rounds[i];
            for j in 0..SUITS {
                let c = (input[i] >> RANKS * j & ((1 << RANKS) - 1)).count_ones();

                p += c * m;
                m *= r + 1;

                r -= c;
            }
        }

        p as usize
    }

    fn colex(&self, s: u64, u: u64) -> u32 {
        let mut x = s as i64;
        let mut i = 0;
        while x > 0 {
            let b = (x & -x) as u64;

            let m = b >> (b - 1 & u).count_ones();

            i |= m;
            x &= x - 1;
        }

        self.colex[i as usize]
    }

    pub fn index(&self, input: SmallVec<[u64; 4]>) -> u32 {
        let p = self.permutation(&input);
        let c = self.map[p];

        let mut a = [(0, 0); 4];
        for i in 0..SUITS as usize {
            let mut x = 0;
            let mut m = 1;

            let mut used: u64 = 0;
            for &cards in &input {
                let cards = cards >> RANKS * i as u32 & ((1 << RANKS) - 1);

                let s = self.nck(RANKS - used.count_ones(), cards.count_ones());

                x += m * self.colex(cards, used);
                m *= s;

                used |= cards;
            }

            a[i] = (self.order[p][i], x)
        }

        println!("{:?} + {}", a, c);

        if a[0] > a[1] {
            a.swap(0, 1);
        }
        if a[2] > a[3] {
            a.swap(2, 3);
        }
        if a[0] > a[2] {
            a.swap(0, 2);
        }
        if a[1] > a[3] {
            a.swap(1, 3);
        }
        if a[1] > a[2] {
            a.swap(1, 2);
        }

        let mut answer = 0;

        let mut cnt = 0;
        let mut sum = 0;
        let mut pre = a[0].0;

        for (i, x) in a {
            if i == pre {
                sum += self.nck(x + cnt, 1 + cnt);
                cnt += 1;
            } else {
                answer =
                    answer * self.nck(self.sizes[c][SUITS as usize - pre] + cnt - 1, cnt) + sum;

                sum = x;
                cnt = 1;
                pre = i;
            }
        }
        answer = answer * self.nck(self.sizes[c][SUITS as usize - pre] + cnt - 1, cnt) + sum;

        answer + self.offsets[c]
    }

    pub fn unindex(&self, mut index: u32) -> SmallVec<[u64; 4]> {
        let mut c = 0;
        let mut b = self.configs.len();
        while c != b {
            let m = (c + b + 1) / 2;

            if self.offsets[m] <= index {
                c = m;
            } else {
                b = m - 1;
            }
        }

        index -= self.offsets[c];

        let mut groups = [0; 4];

        let mut pre = 0;
        let mut cnt = 1;
        for p in 1..=SUITS as usize {
            if p == SUITS as usize || self.configs[c][p] != self.configs[c][pre] {
                let s = self.nck(self.sizes[c][pre] + cnt - 1, cnt);

                let mut i = index % s;

                for k in (p - cnt as usize)..p {
                    let mut idx = 0;
                    let mut bnd = self.sizes[c][pre] - 1;

                    while idx != bnd {
                        let mid = (idx + bnd + 1) / 2;

                        let val = self.nck(mid + (p - k) as u32 - 1, (p - k) as u32);
                        if val <= i {
                            idx = mid;
                        } else {
                            bnd = mid - 1;
                        }
                    }

                    groups[k] = idx;

                    i -= self.nck(idx + (p - k) as u32 - 1, (p - k) as u32);
                }

                cnt = 1;
                pre = p;

                index /= s;
            } else {
                cnt += 1;
            }
        }

        println!("{:?} + {}", groups, c);

        let mut answer = smallvec![0; self.rounds.len()];

        for i in 0..SUITS as usize {
            let mut u = 0u64;
            let mut r = RANKS;

            for j in 0..self.rounds.len() {
                let mut x = 0;

                let n = self.configs[c][i] >> (self.rounds.len() - j - 1) * 4 & 0b1111;
                let s = self.nck(r, n);

                let mut shifted = self.index[n as usize][(groups[i] % s) as usize];
                println!("{}, {}: {:?}", i, j, shifted);

                while shifted > 0 {
                    let p = shifted.trailing_zeros();
                    let b = 1 << p;

                    println!("{}", self.unset[p as usize][u as usize]);

                    x |= 1 << self.unset[p as usize][u as usize];

                    shifted ^= b;
                }

                u |= x;
                r -= n;

                answer[j] |= x << RANKS * i as u32;
                groups[i] /= s;
            }
        }

        answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::smallvec;

    #[test]
    fn test_indexer_size() {
        let indexer = Indexer::new(vec![2, 3, 1, 1]);

        assert_eq!(*indexer.offsets.last().unwrap(), 2484764919);
    }

    #[test]
    fn test_index_simple_small() {
        let indexer = Indexer::new(vec![2]);

        let input = smallvec![1 << 13 | 1];

        let actual = indexer.index(input);
        let expect = 0;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_index_simple_large() {
        let indexer = Indexer::new(vec![2]);

        let input = smallvec![1 << 12 | 1 << 11];

        let actual = indexer.index(input);
        let expect = 168;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_index_1() {
        let indexer = Indexer::new(vec![2, 3]);

        let input = smallvec![1 << 12 | 1, 1 << 4 | 1 << 22 | 1 << 24];

        let actual = indexer.index(input);
        let expect = 1206609;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_index_2() {
        let indexer = Indexer::new(vec![2, 3]);

        let input = smallvec![1 << 4 | 1 << 21, 1 << 5 | 1 << 22 | 1 << 37];

        let actual = indexer.index(input);
        let expect = 602213;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_unindex_small() {
        let indexer = Indexer::new(vec![2]);

        let input = 0;

        let actual: SmallVec<[u64; 4]> = indexer.unindex(input);
        let expect: SmallVec<[u64; 4]> = smallvec![1 << 13 | 1];

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_unindex_large() {
        let indexer = Indexer::new(vec![2]);

        let input = 168;

        let actual: SmallVec<[u64; 4]> = indexer.unindex(input);
        let expect: SmallVec<[u64; 4]> = smallvec![1 << 12 | 1 << 11];

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_unindex_1() {
        let indexer = Indexer::new(vec![2, 3]);

        let input = 1206609;

        let actual: SmallVec<[u64; 4]> = indexer.unindex(input);
        let expect: SmallVec<[u64; 4]> = smallvec![1 << 12 | 1, 1 << 4 | 1 << 22 | 1 << 24];

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_unindex_2() {
        let indexer = Indexer::new(vec![2, 3]);

        let input = 602213;

        let actual: SmallVec<[u64; 4]> = indexer.unindex(input);
        let expect: SmallVec<[u64; 4]> = smallvec![1 << 8 | 1 << 17, 1 << 9 | 1 << 18 | 1 << 37];

        assert_eq!(actual, expect);
    }
}
