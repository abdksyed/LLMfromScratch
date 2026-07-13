# Deployments

Each folder in this directory owns one independently deployed project route on
`syedak.com`.

Current mapping:

| Source folder | Deployment | Public route |
| --- | --- | --- |
| `01-baby-neuron/` | `deployments/baby-neuron/` | `https://syedak.com/baby-neuron/` |
| `02-bpe/` | `deployments/bpe/` | `https://syedak.com/bpe/` |

Deploy one project:

```bash
./scripts/deploy-project.sh baby-neuron
```

To add another project, copy `deployments/baby-neuron/`, update:

- `name`
- `routes[0].pattern`
- `assets.directory`
- `PREFIX` in `worker.js`

Example mapping:

| Source folder | Deployment | Public route |
| --- | --- | --- |
| `02-tokenizer/` | `deployments/tokenizer/` | `https://syedak.com/tokenizer/` |
