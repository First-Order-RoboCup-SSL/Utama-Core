- Running additional computations off main thread for some reason
- Ability to prioritise / wait on only one queue vs waiting on all queues -> Claim is that you have to process both and load is constant so if you fall behind, you are falling behind consistently and we need to fix it separately because it won't work properly.

We claim can't wait on multiple successively i.e. ref then image then ref etc because packet drop or no ref message or lost connection etc - missing other data. Can't skip through.



Single thread - Doesn't support the "long running computation" - Would get less performance if we added it. 
Queues with no blocking - main spins in fast event loop


1 big queue with blocking
  -> Advantage is that you get slightly better latency because the other threads are running and you can put that in another thread.
  -> Kind of depends if they want to do long running other computations.






If the queue is empty then what does main need to do? Is it nothing?


Problems with current approach:
 - Sequential Approach
 - Separation of concerns for the entities and last 5
        -> Enables receivers to have no concept of state which is good as they shouldn't be responsible for knowing what we want to track about the game
 - Threading complexity and potential race conditions  
 - Another advantage is that the controller is only summoned when decision to be amde - no random sleep lengths


-> "Entities Data Structure" -> Plan is just have main manage it after retrieving from the thread safe queue. Only 1 thread; no race conditions. Main give that data to whoever.

Priorities: If you wanted to you could but we think you don't need to because you need to process everything anyway

