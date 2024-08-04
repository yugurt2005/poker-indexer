use std::cmp;

use itertools::Itertools;
use smallvec::SmallVec;

const CHOOSE_SIZE: usize = 15;

const RANKS: u32 = 13;
const SUITS: u32 = 4;

pub struct Indexer {
    rounds: Vec<u32>,

    choose: Vec<Vec<u32>>,

    colex: Vec<u32>,

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

            configs: Vec::new(),
            offsets: Vec::new(),

            sizes: Vec::new(),

            map: vec![0; count],

            order: vec![[0; 4]; count],
        };

        indexer.create_choose();
        indexer.create_colex();
        indexer.enumerate_configs([0; 4], 0);
        indexer.calculate_configs();
        indexer.build_map([0; 4], 0);

        indexer
    }

    fn create_choose(&mut self) {
        self.choose = Vec::new();

        for n in 0..CHOOSE_SIZE {
            self.choose.push(vec![1; n + 1]);
            for k in 1..n {
                self.choose[n][k] = self.choose[n - 1][k - 1] + self.choose[n - 1][k];
            }
        }
    }

    fn create_colex(&mut self) {
        self.colex = vec![0; 1 << RANKS];
        for i in 0u32..1 << RANKS {
            let mut x = i;
            let mut c = 1;
            while c <= i.count_ones() {
                self.colex[i as usize] += self.nck(x.trailing_zeros(), c);

                x &= x - 1;
                c += 1;
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

            let ok = s == self.rounds[round];

            if ok {
                if (1..SUITS).all(|i| config[i as usize] >= config[i as usize - 1]) {
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
        self.offsets.push(0);

        for config in &self.configs {
            // println!("{:?}", config);

            self.sizes.push([0; SUITS as usize]);

            self.offsets.push(1);

            for (key, chunk) in &(0..SUITS as usize).chunk_by(|&x| config[x]) {
                let mut r = RANKS;
                let mut s = 1;
                for k in 0..self.rounds.len() {
                    let c = key >> (self.rounds.len() - k - 1) * 4 & 0b1111;

                    s *= self.nck(r, c);
                    r -= c;
                }

                let v: Vec<usize> = chunk.collect();

                for &x in &v {
                    self.sizes.last_mut().unwrap()[x] = s;
                }

                *self.offsets.last_mut().unwrap() *=
                    self.nck(s + v.len() as u32 - 1, v.len() as u32)
            }
        }

        for i in 1..self.offsets.len() {
            self.offsets[i] += self.offsets[i - 1];
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

            let ok = s == self.rounds[round];

            if ok {
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
                        self.order[p as usize][i] += (permutation[i] > permutation[k]) as usize;
                    }
                }

                let mut sorted = permutation.clone();
                sorted.sort();

                self.map[p as usize] = self.configs.binary_search(&sorted).unwrap();

                // println!("{}: {:?} | {:?}", p, permutation, self.order[p as usize]);

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

    fn colex(&self, mut s: u64, u: u64) -> u32 {
        // TODO: optimize using memoization
        let mut answer = 0;

        let mut i = 1;
        while s > 0 {
            let p = s.trailing_zeros();
            let b = 1 << p;

            let mask = b - 1;
            let rank = p - (mask & u).count_ones();

            answer += self.nck(rank as u32, i);

            s &= s - 1;
            i += 1;
        }

        answer
    }

    fn multicolex(&self, s: SmallVec<[u32; 4]>) -> u32 {
        let mut answer = 0;

        for i in 0..s.len() {
            answer += self.nck(i as u32 + s[i], i as u32 + 1);
        }

        answer
    }

    pub fn index(&self, input: SmallVec<[u64; 4]>) -> u32 {
        let p = self.permutation(&input);
        let c = self.map[p];

        // println!("p: {}, c: {}", p, c);

        let mut a: SmallVec<[(usize, u32); 4]> = (0..SUITS as usize)
            .map(|i| {
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

                (self.order[p][i], x)
            })
            .collect();

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

        // println!("{:?}", a);

        // println!("sizes: {:?}", self.sizes[c]);

        let answer = a
            .into_iter()
            .chunk_by(|x| x.0)
            .into_iter()
            .fold(0, |acc, ele| {
                // println!("{} {:?}", acc, ele.0);
                let v: SmallVec<[u32; 4]> = ele.1.map(|x| x.1).collect();
                acc * self.nck(self.sizes[c][ele.0] + v.len() as u32 - 1, v.len() as u32)
                    + self.multicolex(v)
            });

        println!("answer: {}", answer);
        println!("offsets: {:?}", self.offsets[c]);

        answer + self.offsets[c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::smallvec;

    #[test]
    fn test_indexer_simple_small() {
        let indexer = Indexer::new(vec![2]);

        let input = smallvec![3];

        let actual = indexer.index(input);
        let expect = 0;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_indexer_simple_large() {
        let indexer = Indexer::new(vec![2]);

        let input = smallvec![1 << 12 | 1 << 25];

        let actual = indexer.index(input);
        let expect = 168;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_indexer_1() {
        let indexer = Indexer::new(vec![2, 3]);

        let input = smallvec![1 << 12 | 1, 1 << 4 | 1 << 22 | 1 << 24];

        let actual = indexer.index(input);
        let expect = 123930;

        assert_eq!(actual, expect);
    }

    #[test]
    fn test_indexer_2() {
        let indexer = Indexer::new(vec![2, 3]);

        let input = smallvec![1 << 4 | 1 << 21, 1 << 5 | 1 << 22 | 1 << 37];

        let actual = indexer.index(input);
        let expect = 772331;

        assert_eq!(actual, expect);
    }
}
