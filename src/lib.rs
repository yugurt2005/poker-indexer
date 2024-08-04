use std::cmp;

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

    order: Vec<[u8; SUITS as usize]>,
}

impl Indexer {
    pub fn new(rounds: Vec<u32>) -> Indexer {
        let mut indexer = Indexer {
            rounds,

            choose: Vec::new(),

            colex: Vec::new(),

            configs: vec![[0; 4]],
            offsets: vec![0],

            sizes: vec![[1; 4]],

            map: Vec::new(),

            order: Vec::new(),
        };

        indexer.create_choose();
        indexer.create_colex();

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

        for i in 0..u32::pow(n + 1, SUITS) {
            let mut x = i;
            let mut s = 0;
            for j in 0..SUITS {
                config[j as usize] |= (x % n) << (p * 4);

                s += x % n;
                x /= n;
            }

            let ok = s == self.rounds[round]
                && (1..SUITS).all(|j| config[j as usize] >= config[j as usize - 1]);

            if ok {
                self.configs.push(config);

                self.sizes.push([0; 4]);

                let mut j = 0;
                while j < SUITS {
                    let mut r = RANKS;
                    let mut s = 1;
                    for k in 0..=round {
                        let c = config[j as usize] >> (self.rounds.len() - k - 1) * 4 & 0b1111;

                        s *= self.nck(r, c);
                        r -= c;
                    }

                    let mut k = 0;
                    while k < SUITS as usize && config[j as usize] == config[k] {
                        self.sizes.last_mut().unwrap()[k] = s;
                        k += 1;
                    }

                    j = k as u32;
                }

                self.enumerate_configs(config, round + 1);
            }

            for j in 0..SUITS {
                config[j as usize] >>= p * 4 + 4;
                config[j as usize] <<= p * 4 + 4;
            }
        }

        if round == 0 {
            for i in 1..self.offsets.len() {
                self.offsets[i] += self.offsets[i - 1];
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
                answer *= n - i;
                answer /= i + 1;
            }

            answer
        }
    }

    fn permutation(&self) -> usize {
        0
    }

    fn colex(&self, mut s: u64, u: u64) -> u32 {
        let mut answer = 0;

        let m = s.count_ones();
        for i in 1..=m {
            let p = 5; //s.trailing_zeros();
            let b = 1 << p;

            let mask = b - 1;
            let rank = p - (mask & u).count_ones();

            answer += self.nck(rank as u32, i);

            s &= s - 1;
        }

        answer
    }

    #[inline]
    fn colex_multi_1(&self, a1: u32) -> u32 {
        a1
    }

    #[inline]
    fn colex_multi_2(&self, a1: u32, a2: u32) -> u32 {
        a2 + self.nck(a1 + 1, 2)
    }

    #[inline]
    fn colex_multi_3(&self, a1: u32, a2: u32, a3: u32) -> u32 {
        a3 + self.nck(a2 + 1, 2) + self.nck(a1 + 2, 3)
    }

    #[inline]
    fn colex_multi_4(&self, a1: u32, a2: u32, a3: u32, a4: u32) -> u32 {
        a4 + self.nck(a3 + 1, 2) + self.nck(a2 + 2, 3) + self.nck(a1 + 3, 4)
    }

    pub fn index(&self, input: SmallVec<[u64; 4]>) -> u32 {
        let mut indices = [0; 4];

        for i in 0..SUITS as usize {
            let mut used = 0;
            for &cards in &input {
                let cards = cards >> RANKS * i as u32 & ((1 << RANKS) - 1);

                indices[i] += self.colex(cards, used)
                    * self.nck(RANKS - used.count_ones(), cards.count_ones());

                used |= cards;
            }
        }

        let p = self.permutation();

        let mut answer = 0;

        answer
    }
}
