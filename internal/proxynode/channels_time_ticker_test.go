package proxynode

import (
	"context"
	"math/rand"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"

	"github.com/stretchr/testify/assert"
)

type mockTsoAllocator struct {
}

func (tso *mockTsoAllocator) AllocOne() (Timestamp, error) {
	return Timestamp(time.Now().UnixNano()), nil
}

func newMockTsoAllocator() *mockTsoAllocator {
	return &mockTsoAllocator{}
}

func getStatistics(pchan pChan) (pChanStatistics, error) {
	stats := pChanStatistics{
		minTs:   Timestamp(time.Now().UnixNano()),
		invalid: false,
	}
	stats.maxTs = stats.minTs + Timestamp(time.Millisecond*10)
	return stats, nil
}

func TestChannelsTimeTickerImpl_start(t *testing.T) {
	interval := time.Millisecond * 10
	pchanNum := rand.Uint64()%10 + 1
	pchans := make([]pChan, 0, pchanNum)
	for i := 0; uint64(i) < pchanNum; i++ {
		pchans = append(pchans, genUniqueStr())
	}
	tso := newMockTsoAllocator()
	ctx := context.Background()

	ticker := newChannelsTimeTicker(ctx, interval, pchans, getStatistics, tso)
	err := ticker.start()
	assert.Equal(t, nil, err)

	defer func() {
		err := ticker.close()
		assert.Equal(t, nil, err)
	}()

	time.Sleep(time.Second)
}

func TestChannelsTimeTickerImpl_close(t *testing.T) {
	interval := time.Millisecond * 10
	pchanNum := rand.Uint64()%10 + 1
	pchans := make([]pChan, 0, pchanNum)
	for i := 0; uint64(i) < pchanNum; i++ {
		pchans = append(pchans, genUniqueStr())
	}
	tso := newMockTsoAllocator()
	ctx := context.Background()

	ticker := newChannelsTimeTicker(ctx, interval, pchans, getStatistics, tso)
	err := ticker.start()
	assert.Equal(t, nil, err)

	defer func() {
		err := ticker.close()
		assert.Equal(t, nil, err)
	}()

	time.Sleep(time.Second)
}

func TestChannelsTimeTickerImpl_addPChan(t *testing.T) {
	interval := time.Millisecond * 10
	pchanNum := rand.Uint64()%10 + 1
	pchans := make([]pChan, 0, pchanNum)
	for i := 0; uint64(i) < pchanNum; i++ {
		pchans = append(pchans, genUniqueStr())
	}
	tso := newMockTsoAllocator()
	ctx := context.Background()

	ticker := newChannelsTimeTicker(ctx, interval, pchans, getStatistics, tso)
	err := ticker.start()
	assert.Equal(t, nil, err)

	newPChanNum := rand.Uint64()%10 + 1
	for i := 0; uint64(i) < newPChanNum; i++ {
		err = ticker.addPChan(genUniqueStr())
		assert.Equal(t, nil, err)
	}

	defer func() {
		err := ticker.close()
		assert.Equal(t, nil, err)
	}()

	time.Sleep(time.Second)
}

func TestChannelsTimeTickerImpl_getLastTick(t *testing.T) {
	interval := time.Millisecond * 10
	pchanNum := rand.Uint64()%10 + 1
	pchans := make([]pChan, 0, pchanNum)
	for i := 0; uint64(i) < pchanNum; i++ {
		pchans = append(pchans, genUniqueStr())
	}
	tso := newMockTsoAllocator()
	ctx := context.Background()

	ticker := newChannelsTimeTicker(ctx, interval, pchans, getStatistics, tso)
	err := ticker.start()
	assert.Equal(t, nil, err)

	var wg sync.WaitGroup
	wg.Add(1)
	b := make(chan struct{}, 1)
	go func() {
		defer wg.Done()
		timer := time.NewTicker(interval * 40)
		for {
			select {
			case <-b:
				return
			case <-timer.C:
				for _, pchan := range pchans {
					ts, err := ticker.getLastTick(pchan)
					assert.Equal(t, nil, err)
					log.Debug("TestChannelsTimeTickerImpl_getLastTick",
						zap.Any("pchan", pchan),
						zap.Any("minTs", ts))
				}
			}
		}
	}()
	time.Sleep(time.Second)
	b <- struct{}{}
	wg.Wait()

	defer func() {
		err := ticker.close()
		assert.Equal(t, nil, err)
	}()

	time.Sleep(time.Second)
}